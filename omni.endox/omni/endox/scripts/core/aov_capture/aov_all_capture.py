"""
aov_capture_all.py - Unified Multi-AOV Capture for Omniverse Replicator
========================================================================

Captures all (or selected) AOV modalities in a single end-to-end run
and saves each modality into its own subfolder under ``output_dir``.

Capture runs in up to two phases, because motion vectors require a
different renderer than the path-traced AOVs:
  * Phase 1 (RTX Real-Time): optical_flow only.  Motion vectors are a
    real-time rasterization/GBuffer product and are NOT produced by the
    PathTracing integrator, so flow is captured first in its own pass with
    previous-frame priming.  Skipped if optical_flow is not requested.
  * Phase 2 (PathTracing): rgb, depth, normals, camera_pose - one shared
    render product, one pass over the timeline.

All GPU-capable modalities (depth, normals, optical flow) are accelerated
with NVIDIA Warp - annotator data is fetched directly on CUDA, collected
into GPU buffers over ``write_interval`` frames, batch-converted via Warp
kernels, transferred to CPU once, and written in parallel via a thread pool.

Supported modalities
--------------------
  rgb            : LDR colour image (8-bit RGBA PNG)
  depth          : Path-traced Z-depth (16-bit single-channel PNG)  [GPU]
  normals_world  : World-space surface normals (16-bit BGR PNG)     [GPU]
  normals_camera : Camera-space surface normals (16-bit BGR PNG)    [GPU]
  optical_flow   : Per-pixel motion vectors (16-bit RGB PNG)        [GPU]
  camera_pose    : Camera translation + quaternion per frame (text file)
  occlusion      : Occlusion mask via clipping-plane sweep (8-bit RGBA) [GPU]
                   ** Must run alone - cannot be combined with others **

Output structure
----------------
  <output_dir>/
  ├── rgb/             0000.png  0001.png  ...
  ├── depth/           0000.png  0001.png  ...
  ├── normals_world/   0000.png  0001.png  ...
  ├── normals_camera/  0000.png  0001.png  ...
  ├── optical_flow/    0000.png  0001.png  ...
  └── camera_pose/     camera_pose.txt

Usage (Omniverse Script Editor)
-------------------------------
  import asyncio
  from aov_capture_all import AOVCaptureManager

  manager = AOVCaptureManager(
      camera_prim_path="/World/CapsuleCam/Camera",
      output_dir="C:/output/aov_capture",
      resolution=(512, 512),
      modalities=["rgb", "depth", "normals_camera", "optical_flow", "camera_pose"],
      write_interval=5,
  )
  asyncio.ensure_future(manager.capture_all())

Notes
-----
- ``render_product`` is created **once** and shared by every annotator.
- ``normals_world`` and ``normals_camera`` both use the ``"normals"``
  annotator; it is attached only once and the data is post-processed
  differently for each variant.
- ``camera_pose`` extracts translation + quaternion via USD and does not
  require an annotator.
- **Occlusion** (``modalities=["occlusion"]``) must be specified alone.
  It uses RTX Real-Time mode with a clipping-plane sweep.
- ``coverage_map`` (mesh vertex visibility) is a separate utility --
  see ``coverage_map_warp.py``.
"""

import omni.replicator.core as rep
import omni.usd
import omni.timeline
import asyncio
import numpy as np
import cv2
import carb
import os
import time
import warp as wp
wp.init()
import _warp_compat  # noqa: F401  (patches wp.types.array for replicator compat)
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Import reusable helpers from individual capture scripts
# ---------------------------------------------------------------------------
from depth_capture import DepthCapture
from surface_normal_camera_capture import ViewSurfaceNormalCapture
from surface_normal_world_capture import WorldSurfaceNormalCapture
from optical_flow_capture import OpticalFlowCapture
from camera_pose_capture import CameraPoseCapture
from occlusion_capture import OcclusionCapture


# ---------------------------------------------------------------------------
# AOVCaptureManager
# ---------------------------------------------------------------------------
class AOVCaptureManager:
    """Unified AOV capture manager for Omniverse Replicator.

    Creates a **single** ``render_product`` and attaches all necessary
    annotators to it.  Iterates through the USD timeline exactly once
    and saves each modality into its own subfolder.

    Batched pipeline (per ``write_interval`` frames):
        1. **Collect** - fetch annotator data on CUDA, copy raw channels
           into per-modality f32 GPU buffer slots (async, upcast if f16).
        2. **Rotate** - camera-space normals: per-slot rotation kernel (async).
        3. **Convert** - ONE Warp kernel per modality converts all N raw
           frames -> uint16 output in parallel on the GPU.
        4. **Transfer** - single ``.numpy()`` DMA per modality.
        5. **Write** - all frames x all modalities written via
           ``ThreadPoolExecutor`` in parallel.

    Note: Occlusion (``modalities=["occlusion"]``) must be run alone because
    it requires a dedicated clipping-plane sweep under RTX Real-Time mode.
    When selected, the manager delegates to ``OcclusionCapture`` internally.
    """

    ALL_MODALITIES = [
        "rgb",
        "depth",
        "normals_world",
        "normals_camera",
        "optical_flow",
        "camera_pose",
        "occlusion",
    ]

    # Maps modality -> Replicator annotator name
    _ANNOTATOR_MAP = {
        "rgb": "rgb",
        "depth": "PtZDepth",
        "normals_world": "normals",
        "normals_camera": "normals",       # same annotator, different post-process
        "optical_flow": "motion_vectors",
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        camera_prim_path: str,
        output_dir: str,
        resolution=(512, 512),
        modalities=None,
        samples_per_pixel: int = 512,
        verbose: bool = True,
        write_interval: int = 5,
        num_frames: int = None,
        # Occlusion-specific parameters
        num_clipping_steps: int = 100,
        near: float = 1.0,
        far: float = 100.0,
        near_increment: int = 1,
        diff_threshold: float = 0.05,
        black_eps: float = 0.02,
    ):
        """
        Parameters
        ----------
        camera_prim_path : str
            USD prim path to the camera.
        output_dir : str
            Root directory for all outputs.
        resolution : tuple[int, int]
            ``(width, height)`` of the render product.
        modalities : list[str] | None
            Which modalities to capture.  ``None`` = all non-occlusion
            modalities.  ``["occlusion"]`` must be specified alone.
        samples_per_pixel : int
            Path-tracing SPP for the render settings.
        verbose : bool
            Print progress messages.
        write_interval : int
            Number of frames to batch on GPU before flushing to disk.
        num_frames : int | None
            Number of frames to capture.  ``None`` = all timeline frames.
        num_clipping_steps : int
            (Occlusion only) Number of clipping sweep steps per frame.
        near / far : float
            (Occlusion only) Near and far clip distances.
        near_increment : int
            (Occlusion only) Near-clip step size.
        diff_threshold : float
            (Occlusion only) RGB diff threshold for occlusion detection.
        black_eps : float
            (Occlusion only) Tolerance for near-black pixels.
        """
        # ---- Validate modality selection ----
        if modalities is None:
            modalities = [m for m in self.ALL_MODALITIES if m != "occlusion"]
        unknown = set(modalities) - set(self.ALL_MODALITIES)
        if unknown:
            raise ValueError(
                f"Unknown modalities: {unknown}. "
                f"Choose from {self.ALL_MODALITIES}"
            )

        # Occlusion cannot be mixed with other modalities
        self.is_occlusion = "occlusion" in modalities
        if self.is_occlusion and len(modalities) > 1:
            raise ValueError(
                "Occlusion must run alone - it requires a dedicated "
                "clipping-plane sweep. Use modalities=['occlusion'].")

        self.modalities = list(modalities)

        self._paused = False
        self._stopped = False

        self.camera_prim_path = camera_prim_path
        self.output_dir = output_dir
        self.resolution = resolution
        self.samples_per_pixel = samples_per_pixel
        self.verbose = verbose
        self.write_interval = max(1, write_interval)

        # ---- USD / Timeline ----
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.tcps = self.timeline.get_time_codes_per_seconds()
        self.start_time = self.timeline.get_start_time()
        self.end_time = self.timeline.get_end_time()
        tl_frames = int((self.end_time - self.start_time) * self.tcps)
        self.num_frames = (min(num_frames, tl_frames)
                           if num_frames else tl_frames)
        self.camera_prim = self.stage.GetPrimAtPath(self.camera_prim_path)

        # ---- Occlusion: delegate to OcclusionCapture ----
        if self.is_occlusion:
            self._occlusion = OcclusionCapture(
                camera_prim_path=camera_prim_path,
                output_dir=os.path.join(self.output_dir, "occlusion"),
                resolution=resolution,
                num_clipping_steps=num_clipping_steps,
                near=near,
                far=far,
                near_increment=near_increment,
                diff_threshold=diff_threshold,
                black_eps=black_eps,
                write_interval=write_interval,
                verbose=verbose,
                num_frames=num_frames,
            )
            return  # no further init needed for non-occlusion pipeline

        # ---- Create per-modality output directories ----
        for mod in self.modalities:
            os.makedirs(os.path.join(self.output_dir, mod), exist_ok=True)

        # ---- Render settings ----
        rep.settings.set_render_pathtraced(
            samples_per_pixel=self.samples_per_pixel
        )

        # ---- Warp (GPU-accelerated helpers) ----
        wp.init()
        cuda_idx = len(wp.get_cuda_devices()) - 1
        self.device = f"cuda:{cuda_idx}"

        # ---- Per-modality GPU buffers (lazy-allocated) ----
        self._bufs = {}   # modality -> {"raw": wp.array, "out": wp.array}

    # ------------------------------------------------------------------
    # Annotator helpers
    # ------------------------------------------------------------------
    def _required_annotator_names(self, modalities=None):
        """Return the unique set of annotator names for selected modalities."""
        names = set()
        for mod in (modalities if modalities is not None else self.modalities):
            if mod in self._ANNOTATOR_MAP:
                names.add(self._ANNOTATOR_MAP[mod])
        return names

    # ------------------------------------------------------------------
    # GPU buffer management
    # ------------------------------------------------------------------
    def _ensure_depth_bufs(self, h, w):
        N = self.write_interval
        if "depth" in self._bufs:
            return
        self._bufs["depth"] = {
            "raw": wp.zeros((N, h, w), dtype=wp.float32, device=self.device),
            "out": wp.zeros((N, h, w), dtype=wp.uint16, device=self.device),
        }

    def _ensure_normals_world_bufs(self, h, w):
        N = self.write_interval
        if "normals_world" in self._bufs:
            return
        self._bufs["normals_world"] = {
            "raw": wp.zeros((N, h, w, 3), dtype=wp.float32, device=self.device),
            "out": wp.zeros((N, h, w, 3), dtype=wp.uint16, device=self.device),
        }

    def _ensure_normals_camera_bufs(self, h, w):
        N = self.write_interval
        if "normals_camera" in self._bufs:
            return
        self._bufs["normals_camera"] = {
            "raw": wp.zeros((N, h, w, 3), dtype=wp.float32, device=self.device),
            "out": wp.zeros((N, h, w, 3), dtype=wp.uint16, device=self.device),
        }

    def _ensure_flow_bufs(self, h, w):
        N = self.write_interval
        if "optical_flow" in self._bufs:
            return
        self._bufs["optical_flow"] = {
            "raw": wp.zeros((N, h, w, 2), dtype=wp.float32, device=self.device),
            "out": wp.zeros((N, h, w, 3), dtype=wp.uint16, device=self.device),
        }

    # ------------------------------------------------------------------
    # GPU collect helpers (copy raw data into buffer slot, async)
    # ------------------------------------------------------------------
    def _collect_depth(self, depth_cuda, slot):
        dev = self.device
        h, w = depth_cuda.shape[0], depth_cuda.shape[1]
        self._ensure_depth_bufs(h, w)
        kernel = (DepthCapture._copy_extract_f32_kernel
                  if depth_cuda.dtype == wp.float32
                  else DepthCapture._copy_extract_f16_to_f32_kernel)
        wp.launch(
            kernel=kernel,
            dim=(h, w),
            inputs=[depth_cuda, self._bufs["depth"]["raw"], slot],
            device=dev,
        )

    def _collect_normals_world(self, normals_cuda, slot):
        dev = self.device
        h, w = normals_cuda.shape[0], normals_cuda.shape[1]
        self._ensure_normals_world_bufs(h, w)
        kernel = (WorldSurfaceNormalCapture._copy_extract_normals_f32_kernel
                  if normals_cuda.dtype == wp.float32
                  else WorldSurfaceNormalCapture._copy_extract_normals_f16_to_f32_kernel)
        wp.launch(
            kernel=kernel,
            dim=(h, w),
            inputs=[normals_cuda, self._bufs["normals_world"]["raw"], slot],
            device=dev,
        )

    def _collect_normals_camera(self, normals_cuda, slot, camera_rot):
        dev = self.device
        h, w = normals_cuda.shape[0], normals_cuda.shape[1]
        self._ensure_normals_camera_bufs(h, w)
        # 1. Copy raw normals into slot
        kernel = (ViewSurfaceNormalCapture._copy_extract_normals_f32_kernel
                  if normals_cuda.dtype == wp.float32
                  else ViewSurfaceNormalCapture._copy_extract_normals_f16_to_f32_kernel)
        wp.launch(
            kernel=kernel,
            dim=(h, w),
            inputs=[normals_cuda, self._bufs["normals_camera"]["raw"], slot],
            device=dev,
        )
        # 2. Rotate slot in-place (pass R as wp.array, not scalar floats --
        #    avoids Warp 1.4.1 codegen bug)
        rot_np = camera_rot.astype(np.float32)
        rot_wp = wp.array(rot_np, dtype=wp.float32, device=dev)
        wp.launch(
            kernel=ViewSurfaceNormalCapture._rotate_slot_kernel,
            dim=(h, w),
            inputs=[self._bufs["normals_camera"]["raw"], rot_wp, slot],
            device=dev,
        )

    def _collect_flow(self, flow_cuda, slot):
        dev = self.device
        h, w = flow_cuda.shape[0], flow_cuda.shape[1]
        self._ensure_flow_bufs(h, w)
        kernel = (OpticalFlowCapture._copy_extract_flow_f32_kernel
                  if flow_cuda.dtype == wp.float32
                  else OpticalFlowCapture._copy_extract_flow_f16_to_f32_kernel)
        wp.launch(
            kernel=kernel,
            dim=(h, w),
            inputs=[flow_cuda, self._bufs["optical_flow"]["raw"], slot],
            device=dev,
        )

    # ------------------------------------------------------------------
    # GPU batch convert (all N frames at once per modality)
    # ------------------------------------------------------------------
    def _batch_convert_depth(self, count):
        b = self._bufs["depth"]
        h, w = b["out"].shape[1], b["out"].shape[2]
        wp.launch(
            kernel=DepthCapture._batch_convert_kernel,
            dim=(count, h, w),
            inputs=[b["raw"], b["out"]],
            device=self.device,
        )

    def _batch_convert_normals_world(self, count):
        b = self._bufs["normals_world"]
        h, w = b["out"].shape[1], b["out"].shape[2]
        wp.launch(
            kernel=WorldSurfaceNormalCapture._batch_convert_normals_kernel,
            dim=(count, h, w),
            inputs=[b["raw"], b["out"]],
            device=self.device,
        )

    def _batch_convert_normals_camera(self, count):
        b = self._bufs["normals_camera"]
        h, w = b["out"].shape[1], b["out"].shape[2]
        wp.launch(
            kernel=ViewSurfaceNormalCapture._batch_convert_normals_kernel,
            dim=(count, h, w),
            inputs=[b["raw"], b["out"]],
            device=self.device,
        )

    def _batch_convert_flow(self, count):
        b = self._bufs["optical_flow"]
        h, w = b["out"].shape[1], b["out"].shape[2]
        wp.launch(
            kernel=OpticalFlowCapture._batch_convert_flow_kernel,
            dim=(count, h, w),
            inputs=[b["raw"], b["out"]],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Flush: batch convert -> transfer -> parallel write
    # ------------------------------------------------------------------
    def _flush(self, count, batch_paths, batch_rgb, batch_pose_lines,
               pose_file):
        """
        Parameters
        ----------
        count : int
            Number of frames in this batch.
        batch_paths : dict[str, list[str]]
            Modality -> list of output paths for this batch.
        batch_rgb : list[np.ndarray]
            Collected RGB data (CPU, RGBA arrays).
        batch_pose_lines : list[str]
            Collected camera pose lines.
        pose_file : file handle or None
        """
        if count == 0:
            return

        t0 = time.perf_counter()

        # 1. Batch-convert all GPU modalities (async kernel launches)
        if "depth" in batch_paths and batch_paths["depth"]:
            self._batch_convert_depth(count)
        if "normals_world" in batch_paths and batch_paths["normals_world"]:
            self._batch_convert_normals_world(count)
        if "normals_camera" in batch_paths and batch_paths["normals_camera"]:
            self._batch_convert_normals_camera(count)
        if "optical_flow" in batch_paths and batch_paths["optical_flow"]:
            self._batch_convert_flow(count)

        # 2. Transfer GPU -> CPU (single .numpy() per modality)
        np_data = {}
        if "depth" in batch_paths and batch_paths["depth"]:
            np_data["depth"] = self._bufs["depth"]["out"].numpy()
        if "normals_world" in batch_paths and batch_paths["normals_world"]:
            np_data["normals_world"] = self._bufs["normals_world"]["out"].numpy()
        if "normals_camera" in batch_paths and batch_paths["normals_camera"]:
            np_data["normals_camera"] = self._bufs["normals_camera"]["out"].numpy()
        if "optical_flow" in batch_paths and batch_paths["optical_flow"]:
            np_data["optical_flow"] = self._bufs["optical_flow"]["out"].numpy()

        # 3. Write all frames x all modalities in parallel
        write_tasks = []
        for mod in ("depth", "normals_world", "normals_camera", "optical_flow"):
            if mod not in np_data:
                continue
            for i, path in enumerate(batch_paths[mod]):
                write_tasks.append(("cv2", np_data[mod][i], path))

        for i, path in enumerate(batch_paths.get("rgb", [])):
            write_tasks.append(("pil", batch_rgb[i], path))

        if write_tasks:
            with ThreadPoolExecutor(
                    max_workers=min(len(write_tasks), 16)) as pool:
                futures = []
                for kind, data, path in write_tasks:
                    if kind == "cv2":
                        futures.append(pool.submit(cv2.imwrite, path, data))
                    elif kind == "pil":
                        futures.append(
                            pool.submit(self._pil_save_rgba, data, path))
                for f in futures:
                    f.result()

        # 4. Write camera pose lines
        if pose_file is not None and batch_pose_lines:
            for line in batch_pose_lines:
                pose_file.write(line)

        if self.verbose:
            t_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [flush] {count} frame(s) written  ({t_ms:.1f} ms)")

    # ------------------------------------------------------------------
    # Pause / Stop controls
    # ------------------------------------------------------------------
    def pause(self):
        self._paused = True
        if self.is_occlusion:
            self._occlusion.pause()
        if self.verbose:
            print("[AOVCapture] Paused")

    def resume(self):
        self._paused = False
        if self.is_occlusion:
            self._occlusion.resume()
        if self.verbose:
            print("[AOVCapture] Resumed")

    def stop(self):
        self._stopped = True
        self._paused = False
        if self.is_occlusion:
            self._occlusion.stop()
        if self.verbose:
            print("[AOVCapture] Stop requested")

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    # ------------------------------------------------------------------
    # Main capture loop
    # ------------------------------------------------------------------
    async def capture_all(self):
        """Run the full end-to-end capture for every selected modality.

        This is the single entry point you call via
        ``asyncio.ensure_future(manager.capture_all())``.
        """
        if self.is_occlusion:
            return await self._occlusion.capture_and_visualize()

        if self.num_frames == 0:
            print("[AOVCapture] No frames to capture (timeline is empty).")
            return

        if self.verbose:
            print("=" * 60)
            print("  AOV Capture - GPU-Accelerated Batched Pipeline")
            print("=" * 60)
            print(f"  Camera         : {self.camera_prim_path}")
            print(f"  Resolution     : {self.resolution}")
            print(f"  SPP            : {self.samples_per_pixel}")
            print(f"  Frames         : {self.num_frames}")
            print(f"  Write interval : {self.write_interval}")
            print(f"  Modalities     : {self.modalities}")
            print(f"  Output dir     : {self.output_dir}")
            print(f"  GPU device     : {self.device}")
            print("=" * 60)

        # Capture runs in two passes because motion vectors and the path-traced
        # AOVs need different renderers:
        #   Phase 1 (RTX Real-Time): optical_flow.  Motion vectors are a
        #     real-time rasterization/GBuffer product and are NOT produced by
        #     the PathTracing integrator.
        #   Phase 2 (PathTracing): every other ("stateless") modality.
        needs_flow = "optical_flow" in self.modalities
        pt_mods = [m for m in self.modalities if m != "optical_flow"]

        settings = carb.settings.get_settings()
        original_mode = settings.get("/rtx/rendermode")
        t_total_start = time.perf_counter()

        # ============================================================
        # Phase 1: optical flow in RTX Real-Time (separate render mode)
        # ============================================================
        if needs_flow and not self._stopped:
            await self._capture_flow_realtime()

        # ============================================================
        # Phase 2: PathTracing pass (all modalities except optical_flow)
        # ============================================================
        # Create ONE render product (shared across all annotators)
        render_product = rep.create.render_product(
            self.camera_prim_path, self.resolution
        )
        if self.verbose:
            print("[AOVCapture] Created render product (shared)")

        # ---- Attach annotators (each unique name attached once) ----
        annotators = {}
        for ann_name in self._required_annotator_names(pt_mods):
            ann = rep.AnnotatorRegistry.get_annotator(ann_name)
            ann.attach(render_product)
            annotators[ann_name] = ann
            if self.verbose:
                print(f"[AOVCapture] Attached annotator: {ann_name}")

        # ---- Prepare camera-pose output file ----
        pose_file = None
        if "camera_pose" in self.modalities:
            pose_path = os.path.join(
                self.output_dir, "camera_pose", "camera_pose.txt"
            )
            pose_file = open(pose_path, "w")

        # ---- Disable capture-on-play (we step manually) ----
        rep.orchestrator.set_capture_on_play(False)

        # ---- Timeline setup ----
        start_tc = int(round(self.start_time * self.tcps))

        # Ensure Path Tracing for the path-traced AOVs.  (After Phase 1 the
        # mode was restored to original_mode, which may be RTX Real-Time.)
        current_mode = settings.get("/rtx/rendermode")
        if current_mode != "PathTracing":
            print(f"[AOVCapture] Switching renderer: {current_mode} "
                  f"-> PathTracing")
            settings.set("/rtx/rendermode", "PathTracing")
            await omni.kit.app.get_app().next_update_async()

        if self.verbose:
            print(f"[AOVCapture] Starting capture loop ...")

        # Pre-check which modalities are active
        need_normals = (
            "normals_world" in self.modalities
            or "normals_camera" in self.modalities
        )

        # Batch state
        slot = 0
        batch_paths = {mod: [] for mod in self.modalities}
        batch_rgb = []
        batch_pose_lines = []
        render_total_ms = 0.0

        # ============================================================
        # Per-frame loop (skipped entirely if only optical_flow is requested,
        # since that was handled by the Phase 1 Real-Time pass above)
        # ============================================================
        captured_frames = 0
        pt_frames = self.num_frames if pt_mods else 0
        for timecode in range(start_tc, start_tc + pt_frames):
            # ---- Pause / Stop ----
            while self._paused and not self._stopped:
                await asyncio.sleep(0.1)
            if self._stopped:
                if self.verbose:
                    print(f"[AOVCapture] Stopped at frame {timecode}")
                break

            # Set timeline to the current timecode
            self.timeline.set_current_time(timecode / self.tcps)
            t0_render = time.perf_counter()
            await omni.kit.app.get_app().next_update_async()
            await rep.orchestrator.step_async()
            render_total_ms += (time.perf_counter() - t0_render) * 1e3

            fname = f"{timecode:04d}.png"

            # --- Fetch normals on CUDA once (shared by world & camera) ---
            normals_cuda = None
            if need_normals:
                normals_cuda = annotators["normals"].get_data(
                    device=self.device)

            # --- RGB (CPU - PIL) ---
            if "rgb" in self.modalities:
                rgb_data = annotators["rgb"].get_data()
                batch_rgb.append(rgb_data)
                batch_paths["rgb"].append(
                    os.path.join(self.output_dir, "rgb", fname))

            # --- Depth (GPU collect) ---
            if "depth" in self.modalities:
                depth_cuda = annotators["PtZDepth"].get_data(
                    device=self.device)
                self._collect_depth(depth_cuda, slot)
                batch_paths["depth"].append(
                    os.path.join(self.output_dir, "depth", fname))

            # --- Normals world (GPU collect) ---
            if "normals_world" in self.modalities:
                self._collect_normals_world(normals_cuda, slot)
                batch_paths["normals_world"].append(
                    os.path.join(self.output_dir, "normals_world", fname))

            # --- Normals camera (GPU collect + rotate) ---
            if "normals_camera" in self.modalities:
                camera_rot = ViewSurfaceNormalCapture.get_camera_rotation_matrix(
                    self.camera_prim)
                self._collect_normals_camera(normals_cuda, slot, camera_rot)
                batch_paths["normals_camera"].append(
                    os.path.join(self.output_dir, "normals_camera", fname))

            # --- Optical flow was handled in Phase 1 (RTX Real-Time) ---

            # --- Camera pose (CPU text) ---
            if "camera_pose" in self.modalities:
                tx, ty, tz, qx, qy, qz, qw = \
                    CameraPoseCapture.extract_pose_at_time(self.camera_prim)
                batch_pose_lines.append(
                    f"{tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

            slot += 1
            captured_frames += 1

            # --- Flush at write_interval or last frame ---
            is_last = (timecode == start_tc + self.num_frames - 1)
            if slot >= self.write_interval or is_last or self._stopped:
                self._flush(slot, batch_paths, batch_rgb,
                            batch_pose_lines, pose_file)

                if self.verbose:
                    ftc = timecode - slot + 1
                    print(f"[AOVCapture] Frames {ftc:04d}-{timecode:04d} done")

                # Reset batch state
                slot = 0
                batch_paths = {mod: [] for mod in self.modalities}
                batch_rgb.clear()
                batch_pose_lines.clear()

        # ============================================================
        # Phase 2 cleanup
        # ============================================================
        if pose_file is not None:
            pose_file.close()

        for ann in annotators.values():
            ann.detach(render_product)
        render_product.destroy()

        t_total = time.perf_counter() - t_total_start
        avg_ms = (t_total / max(captured_frames, 1)) * 1000.0

        # Restore original render mode
        if original_mode and original_mode != "PathTracing":
            print(f"[AOVCapture] Restoring renderer: PathTracing "
                  f"-> {original_mode}")
            settings.set("/rtx/rendermode", original_mode)

        render_s = render_total_ms / 1e3
        proc_s = t_total - render_s  # derived: wall-clock minus render
        render_pct = (render_s / t_total * 100) if t_total > 0 else 0

        status = "stopped" if self._stopped else "complete"
        if self.verbose:
            print("=" * 60)
            print(
                f"[AOVCapture] Capture {status} - "
                f"{captured_frames}/{self.num_frames} frames"
            )
            print(f"  Total time      : {t_total:.2f} s  "
                  f"(avg {t_total / max(captured_frames, 1) * 1e3:.2f} ms/frame)")
            print(f"    Render        : {render_s:.2f} s  "
                  f"({render_pct:.1f}%)")
            print(f"    Post-proc+I/O : {proc_s:.2f} s  "
                  f"({100 - render_pct:.1f}%)")
            print("  Folder layout:")
            for mod in self.modalities:
                mod_dir = os.path.join(self.output_dir, mod)
                n_files = len(os.listdir(mod_dir))
                print(f"    {mod + '/':20s} {n_files} files")
            print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: optical flow (RTX Real-Time)
    # ------------------------------------------------------------------
    async def _capture_flow_realtime(self):
        """Dedicated optical-flow pass in RTX Real-Time mode.

        Motion vectors are produced by the real-time rasterization/GBuffer
        pipeline (the renderer needs them for TAA/DLSS reprojection).  The
        PathTracing integrator does NOT populate them, so capturing flow in
        the main PathTracing pass yields all-zero output.  We therefore run
        flow as a separate pass: switch to RTX Real-Time, iterate the timeline
        with previous-frame priming (so the differential motion buffer is
        valid), write the flow PNGs, then restore the previous render mode.
        """
        settings = carb.settings.get_settings()
        prev_mode = settings.get("/rtx/rendermode")

        if self.verbose:
            print("[AOVCapture] --- Phase 1: optical flow (RTX Real-Time) ---")

        # Switch to RTX Real-Time so the motion-vector buffer is generated
        if prev_mode != "RaytracedLighting":
            settings.set("/rtx/rendermode", "RaytracedLighting")
            await omni.kit.app.get_app().next_update_async()

        render_product = rep.create.render_product(
            self.camera_prim_path, self.resolution)
        motion_annot = rep.AnnotatorRegistry.get_annotator("motion_vectors")
        motion_annot.attach(render_product)
        rep.orchestrator.set_capture_on_play(False)

        start_tc = int(round(self.start_time * self.tcps))
        flow_dir = os.path.join(self.output_dir, "optical_flow")

        slot = 0
        batch_paths = {"optical_flow": []}
        for timecode in range(start_tc, start_tc + self.num_frames):
            while self._paused and not self._stopped:
                await asyncio.sleep(0.1)
            if self._stopped:
                break

            # Prime the previous frame, then capture the current one so the
            # differential motion buffer reflects t_{n-1} -> t_n (not 0).
            prev_tc = max(start_tc, timecode - 1)
            self.timeline.set_current_time(prev_tc / self.tcps)
            await omni.kit.app.get_app().next_update_async()
            self.timeline.set_current_time(timecode / self.tcps)
            await rep.orchestrator.step_async()

            flow_cuda = motion_annot.get_data(device=self.device)
            self._collect_flow(flow_cuda, slot)
            batch_paths["optical_flow"].append(
                os.path.join(flow_dir, f"{timecode:04d}.png"))
            slot += 1

            is_last = (timecode == start_tc + self.num_frames - 1)
            if slot >= self.write_interval or is_last or self._stopped:
                self._flush(slot, batch_paths, [], [], None)
                slot = 0
                batch_paths = {"optical_flow": []}

        motion_annot.detach(render_product)
        render_product.destroy()

        # Restore the render mode that was active before this pass
        if prev_mode and prev_mode != "RaytracedLighting":
            settings.set("/rtx/rendermode", prev_mode)
            await omni.kit.app.get_app().next_update_async()

        if self.verbose:
            print("[AOVCapture] --- Phase 1 done ---")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _pil_save_rgba(data, path):
        """Thread-safe helper for PIL RGBA save."""
        img = Image.fromarray(data, "RGBA")
        img.save(path)


# -----------------------------------------------------------------------
# Convenience entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":

    # ----- EDIT THESE PARAMETERS -----
    CAMERA_PRIM_PATH = "/World/CapsuleCam/Camera"
    OUTPUT_DIR = "C:/output/aov_capture/all_aovs"
    RESOLUTION = (512, 512)

    # Select which modalities to capture (comment/uncomment as needed):
    MODALITIES = [
        "rgb",
        "depth",
        "normals_world",
        "normals_camera",
        "optical_flow",
        "camera_pose",
    ]
    # ----------------------------------

    manager = AOVCaptureManager(
        camera_prim_path=CAMERA_PRIM_PATH,
        output_dir=OUTPUT_DIR,
        resolution=RESOLUTION,
        modalities=MODALITIES,
        samples_per_pixel=512,
        verbose=True,
        write_interval=5,
    )
    asyncio.ensure_future(manager.capture_all())
