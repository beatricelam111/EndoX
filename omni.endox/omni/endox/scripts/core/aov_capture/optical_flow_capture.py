import omni.replicator.core as rep
import omni.usd
import omni.timeline
import asyncio
import numpy as np
import cv2
import carb
import warp as wp
wp.init()
import _warp_compat  # noqa: F401  (patches wp.types.array for replicator compat)
import os
import time
from concurrent.futures import ThreadPoolExecutor


class OpticalFlowCapture:
    """Capture optical-flow (motion vectors) using Warp-accelerated GPU batch pipeline.

    Pipeline (per batch of ``write_interval`` frames):
        1. **Collect** - ``get_data(device=cuda)`` -> extract X/Y channels
           into f32 GPU buffer slot (async, no conversion).
        2. **Convert** - one Warp kernel converts ALL raw f32 frames -> uint16
           BGR in parallel on the GPU.
        3. **Transfer + Write** - single ``.numpy()`` DMA of the whole
           ``(N, H, W, 3)`` output buffer -> parallel ``cv2.imwrite`` per frame.

    Parameters
    ----------
    output_format : ``"png"`` | ``"npz"``
        ``"png"`` - 16-bit BGR PNG via ``cv2.imwrite`` (parallel).
        ``"npz"`` - raw uint16 numpy archive via ``np.save`` (fastest).
    """

    def __init__(
        self,
        camera_prim_path,
        output_dir,
        resolution=(512, 512),
        verbose=False,
        write_interval=1,
        output_format="png",
    ):
        self.camera_prim_path = camera_prim_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.resolution = resolution
        self.verbose = verbose
        self.write_interval = max(1, write_interval)
        self.output_format = output_format.lower()
        if self.output_format not in ("png", "npz"):
            raise ValueError(
                f"output_format must be 'png' or 'npz', got '{output_format}'")

        # USD and Timeline context
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.tcps = self.timeline.get_time_codes_per_seconds()
        self.start_time = self.timeline.get_start_time()
        self.end_time = self.timeline.get_end_time()
        self.num_timeline_frames = int(
            (self.end_time - self.start_time) * self.tcps)

        # Warp setup
        wp.init()
        cuda_idx = len(wp.get_cuda_devices()) - 1
        self.device = f"cuda:{cuda_idx}"

    # ---- Warp kernels (f32 pipeline, with f16-input variants) ----------------

    # - Single-frame: f32 input -> uint16 BGR --
    @staticmethod
    @wp.kernel
    def _flow_f32_to_uint16_kernel(
        flow_in: wp.array(dtype=wp.float32, ndim=3),
        flow_out: wp.array(dtype=wp.uint16, ndim=3),
    ):
        i, j = wp.tid()
        vx = (flow_in[i, j, 0] + 512.0) / 1024.0 * 65535.0
        vy = (flow_in[i, j, 1] + 512.0) / 1024.0 * 65535.0
        if vx < 0.0:
            vx = 0.0
        if vx > 65535.0:
            vx = 65535.0
        if vy < 0.0:
            vy = 0.0
        if vy > 65535.0:
            vy = 65535.0
        flow_out[i, j, 0] = wp.uint16(0)
        flow_out[i, j, 1] = wp.uint16(vy)
        flow_out[i, j, 2] = wp.uint16(vx)

    # - Single-frame: f16 input -> uint16 BGR --
    @staticmethod
    @wp.kernel
    def _flow_f16_to_uint16_kernel(
        flow_in: wp.array(dtype=wp.float16, ndim=3),
        flow_out: wp.array(dtype=wp.uint16, ndim=3),
    ):
        i, j = wp.tid()
        vx = (wp.float32(flow_in[i, j, 0]) + 512.0) / 1024.0 * 65535.0
        vy = (wp.float32(flow_in[i, j, 1]) + 512.0) / 1024.0 * 65535.0
        if vx < 0.0:
            vx = 0.0
        if vx > 65535.0:
            vx = 65535.0
        if vy < 0.0:
            vy = 0.0
        if vy > 65535.0:
            vy = 65535.0
        flow_out[i, j, 0] = wp.uint16(0)
        flow_out[i, j, 1] = wp.uint16(vy)
        flow_out[i, j, 2] = wp.uint16(vx)

    # ---- Copy-to-slot kernels (extract X/Y into f32 buffer) ------------------

    @staticmethod
    @wp.kernel
    def _copy_extract_flow_f32_kernel(
        src: wp.array(dtype=wp.float32, ndim=3),
        dst: wp.array(dtype=wp.float32, ndim=4),
        slot: int,
    ):
        i, j = wp.tid()
        dst[slot, i, j, 0] = src[i, j, 0]
        dst[slot, i, j, 1] = src[i, j, 1]

    @staticmethod
    @wp.kernel
    def _copy_extract_flow_f16_to_f32_kernel(
        src: wp.array(dtype=wp.float16, ndim=3),
        dst: wp.array(dtype=wp.float32, ndim=4),
        slot: int,
    ):
        i, j = wp.tid()
        dst[slot, i, j, 0] = wp.float32(src[i, j, 0])
        dst[slot, i, j, 1] = wp.float32(src[i, j, 1])

    # ---- Batch conversion kernel (f32 -> uint16 BGR) -------------------------

    @staticmethod
    @wp.kernel
    def _batch_convert_flow_kernel(
        raw_buf: wp.array(dtype=wp.float32, ndim=4),
        out_buf: wp.array(dtype=wp.uint16, ndim=4),
    ):
        n, i, j = wp.tid()
        vx = (raw_buf[n, i, j, 0] + 512.0) / 1024.0 * 65535.0
        vy = (raw_buf[n, i, j, 1] + 512.0) / 1024.0 * 65535.0
        if vx < 0.0:
            vx = 0.0
        if vx > 65535.0:
            vx = 65535.0
        if vy < 0.0:
            vy = 0.0
        if vy > 65535.0:
            vy = 65535.0
        out_buf[n, i, j, 0] = wp.uint16(0)
        out_buf[n, i, j, 1] = wp.uint16(vy)
        out_buf[n, i, j, 2] = wp.uint16(vx)

    # ---- Single-frame visualisation (also used by aov_capture_all.py) --------

    @staticmethod
    def visualize_motion(motion, out_path, verbose=False):
        """Convert motion vectors to 16-bit BGR PNG and write to disk.

        Accepts either a ``wp.array`` on CUDA (processed via Warp kernel)
        or a ``np.ndarray`` (CPU fallback).
        """
        t0 = time.perf_counter()

        if isinstance(motion, wp.array) and "cuda" in str(motion.device):
            dev = str(motion.device)
            h, w = motion.shape[0], motion.shape[1]
            out_wp = wp.zeros((h, w, 3), dtype=wp.uint16, device=dev)

            if motion.ndim == 3:
                kernel = (OpticalFlowCapture._flow_f32_to_uint16_kernel
                          if motion.dtype == wp.float32
                          else OpticalFlowCapture._flow_f16_to_uint16_kernel)
                wp.launch(
                    kernel=kernel,
                    dim=(h, w), inputs=[motion, out_wp], device=dev,
                )
            else:
                motion_np = motion.numpy()
                OpticalFlowCapture._visualize_motion_numpy(
                    motion_np, out_path)
                return

            bgr_16bit = out_wp.numpy()
        else:
            motion = (motion if isinstance(motion, np.ndarray)
                      else np.array(motion))
            mv_x = motion[..., 0]
            mv_y = motion[..., 1]
            mv_x_scaled = np.clip(
                (mv_x + 512) / 1024 * 65535, 0, 65535).astype(np.uint16)
            mv_y_scaled = np.clip(
                (mv_y + 512) / 1024 * 65535, 0, 65535).astype(np.uint16)
            h, w = mv_x.shape
            bgr_16bit = np.zeros((h, w, 3), dtype=np.uint16)
            bgr_16bit[..., 2] = mv_x_scaled    # R = X
            bgr_16bit[..., 1] = mv_y_scaled    # G = Y

        cv2.imwrite(out_path, bgr_16bit)

        if verbose:
            t_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [flow] {t_ms:.2f} ms  {out_path}")

    @staticmethod
    def _visualize_motion_numpy(motion_np, out_path):
        """Pure-numpy fallback for unexpected array layouts."""
        mv_x = motion_np[..., 0]
        mv_y = motion_np[..., 1]
        mv_x_scaled = np.clip(
            (mv_x + 512) / 1024 * 65535, 0, 65535).astype(np.uint16)
        mv_y_scaled = np.clip(
            (mv_y + 512) / 1024 * 65535, 0, 65535).astype(np.uint16)
        h, w = mv_x.shape
        bgr_16bit = np.zeros((h, w, 3), dtype=np.uint16)
        bgr_16bit[..., 2] = mv_x_scaled
        bgr_16bit[..., 1] = mv_y_scaled
        cv2.imwrite(out_path, bgr_16bit)

    # ---- Capture loop --------------------------------------------------------

    async def capture_and_visualize(self):
        # Motion vectors are a real-time rasterization/GBuffer product; the
        # PathTracing integrator does NOT populate them (flow comes back all
        # zero).  Force RTX Real-Time for this capture and restore afterwards.
        settings = carb.settings.get_settings()
        prev_mode = settings.get("/rtx/rendermode")
        if prev_mode != "RaytracedLighting":
            print(f"[OpticalFlowCapture] Switching renderer: {prev_mode} "
                  f"-> RaytracedLighting (RTX Real-Time)")
            settings.set("/rtx/rendermode", "RaytracedLighting")
            await omni.kit.app.get_app().next_update_async()

        render_product = rep.create.render_product(
            self.camera_prim_path, self.resolution)
        motion_annot = rep.AnnotatorRegistry.get_annotator("motion_vectors")
        motion_annot.attach(render_product)

        start_tc = int(round(self.start_time * self.tcps))

        print(f"[OpticalFlowCapture] Starting - "
              f"{self.num_timeline_frames} frames, "
              f"write_interval={self.write_interval}, "
              f"format={self.output_format}")

        t_total_start = time.perf_counter()

        # Batch buffer state - lazy-allocated on first frame
        self._raw_buf = None
        self._out_buf = None
        batch_paths = []
        slot = 0

        for timecode in range(start_tc, start_tc + self.num_timeline_frames):
            # ---- Prime previous frame, then capture (differential flow) -------
            # Motion vectors are the screen-space delta between the captured
            # render and the render immediately preceding it.  Rendering the
            # PREVIOUS timecode first seeds the renderer's previous-frame
            # transform cache, so the captured render at the current timecode
            # yields motion for t_{n-1} -> t_n instead of t_n -> t_n (== 0).
            prev_tc = max(start_tc, timecode - 1)
            self.timeline.set_current_time(prev_tc / self.tcps)
            await omni.kit.app.get_app().next_update_async()
            self.timeline.set_current_time(timecode / self.tcps)
            await rep.orchestrator.step_async()

            ext = "png" if self.output_format == "png" else "npz"
            out_path = os.path.join(
                self.output_dir, f'motion_{timecode:04d}.{ext}')

            # ---- Phase 1: Collect - get data + copy raw into slot (async) ----
            flow_cuda = motion_annot.get_data(device=self.device)
            h, w = flow_cuda.shape[0], flow_cuda.shape[1]
            self._ensure_buffers(h, w)

            self._copy_to_raw_slot(flow_cuda, slot)
            batch_paths.append(out_path)
            slot += 1

            # ---- Phase 2 & 3: Convert + Transfer + Write (at interval) ------
            is_last = (timecode == start_tc + self.num_timeline_frames - 1)
            if slot >= self.write_interval or is_last:
                self._flush_buffer(batch_paths, slot)
                batch_paths.clear()
                slot = 0

        t_total = time.perf_counter() - t_total_start
        avg_ms = (t_total / max(self.num_timeline_frames, 1)) * 1000.0

        motion_annot.detach(render_product)
        render_product.destroy()

        # Restore the render mode that was active before capture
        if prev_mode and prev_mode != "RaytracedLighting":
            settings.set("/rtx/rendermode", prev_mode)
            await omni.kit.app.get_app().next_update_async()

        print(f"\n[OpticalFlowCapture] Done - "
              f"{self.num_timeline_frames} frames in "
              f"{t_total:.2f} s  (avg {avg_ms:.2f} ms/frame)")

    # ---- Internal helpers ----------------------------------------------------

    def _ensure_buffers(self, h, w):
        """Lazy-allocate raw input buffer (N,H,W,2) f32 and output (N,H,W,3) uint16."""
        N = self.write_interval
        if self._out_buf is not None and self._out_buf.shape[1:3] == (h, w):
            return
        self._out_buf = wp.zeros((N, h, w, 3), dtype=wp.uint16,
                                 device=self.device)
        self._raw_buf = wp.zeros((N, h, w, 2), dtype=wp.float32,
                                 device=self.device)

    def _copy_to_raw_slot(self, flow_cuda, slot):
        """Extract X,Y channels into ``_raw_buf[slot]`` (async).

        Handles both float32 and float16 annotator outputs.
        """
        h, w = flow_cuda.shape[0], flow_cuda.shape[1]
        dev = str(flow_cuda.device)
        kernel = (OpticalFlowCapture._copy_extract_flow_f32_kernel
                  if flow_cuda.dtype == wp.float32
                  else OpticalFlowCapture._copy_extract_flow_f16_to_f32_kernel)
        wp.launch(
            kernel=kernel,
            dim=(h, w),
            inputs=[flow_cuda, self._raw_buf, slot],
            device=dev,
        )

    def _batch_convert(self, count):
        """Convert ``count`` raw flow frames -> uint16 BGR in one kernel."""
        h, w = self._out_buf.shape[1], self._out_buf.shape[2]
        wp.launch(
            kernel=OpticalFlowCapture._batch_convert_flow_kernel,
            dim=(count, h, w),
            inputs=[self._raw_buf, self._out_buf],
            device=self.device,
        )

    def _flush_buffer(self, paths, count):
        """Batch-convert -> single .numpy() -> write (PNG parallel or NPZ)."""
        if count == 0:
            return
        t0 = time.perf_counter()

        # 1. Convert all raw frames -> uint16 BGR (one kernel launch)
        self._batch_convert(count)

        # 2. One GPU->CPU transfer for the entire batch
        batch_np = self._out_buf.numpy()    # (N, H, W, 3)

        # 3. Write frames to disk
        if self.output_format == "png":
            with ThreadPoolExecutor(max_workers=count) as pool:
                futures = [
                    pool.submit(cv2.imwrite, paths[i], batch_np[i])
                    for i in range(count)
                ]
                for f in futures:
                    f.result()
        else:
            for i in range(count):
                np.save(paths[i], batch_np[i])

        if self.verbose:
            t_ms = (time.perf_counter() - t0) * 1000.0
            first = os.path.basename(paths[0])
            last = os.path.basename(paths[count - 1])
            fmt = "PNG parallel" if self.output_format == "png" else "NPZ"
            print(f"  [flow] Flushed {count} frame(s): "
                  f"{first} -> {last}  ({t_ms:.2f} ms, {fmt})")


if __name__ == "__main__":
    camera_prim_path = "/World/CapsuleCam/Camera"
    output_dir = "C:/output/aov_capture/optical_flow"
    flow_capturer = OpticalFlowCapture(
        camera_prim_path=camera_prim_path,
        output_dir=output_dir,
        resolution=(512, 512),
        verbose=True,
        write_interval=5,
    )
    asyncio.ensure_future(flow_capturer.capture_and_visualize())
