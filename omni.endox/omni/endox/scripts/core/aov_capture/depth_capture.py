import omni.replicator.core as rep
import omni.usd
import omni.timeline
import asyncio
import numpy as np
import cv2
import warp as wp
wp.init()
import _warp_compat  # noqa: F401  (patches wp.types.array for replicator compat)
import os
import time
from concurrent.futures import ThreadPoolExecutor


class DepthCapture:
    """Capture depth maps using Warp-accelerated GPU batch pipeline.

    Pipeline (per batch of ``write_interval`` frames):
        1. **Collect** - ``get_data(device=cuda)`` -> extract depth channel
           into GPU buffer slot (async, no conversion).
        2. **Convert** - one Warp kernel converts ALL raw frames -> uint16
           greyscale in parallel on the GPU.
        3. **Transfer + Write** - single ``.numpy()`` DMA of the whole
           ``(N, H, W)`` output buffer -> parallel ``cv2.imwrite`` (PNG) or
           ``np.save`` (NPZ) per frame.

    Parameters
    ----------
    output_format : ``"png"`` | ``"npz"``
        ``"png"`` - 16-bit greyscale PNG via ``cv2.imwrite`` (parallel).
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

        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.tcps = self.timeline.get_time_codes_per_seconds()
        self.start_time = self.timeline.get_start_time()
        self.end_time = self.timeline.get_end_time()
        self.num_timeline_frames = int(
            (self.end_time - self.start_time) * self.tcps)

        self.resolution = resolution
        self.stage_camera = self.stage.GetPrimAtPath(self.camera_prim_path)
        self.clipping_range_attr = self.stage_camera.GetAttribute(
            "clippingRange")
        self.verbose = verbose
        self.write_interval = max(1, write_interval)
        self.output_format = output_format.lower()
        if self.output_format not in ("png", "npz"):
            raise ValueError(
                f"output_format must be 'png' or 'npz', got '{output_format}'")

        rep.settings.set_render_pathtraced(samples_per_pixel=512)

        wp.init()
        cuda_idx = len(wp.get_cuda_devices()) - 1
        self.device = f"cuda:{cuda_idx}"

    # ---- Warp kernels (float32 pipeline, with f16-input variants) ---------------

    @staticmethod
    @wp.kernel
    def _depth_f32_to_uint16_kernel(
        depth_in: wp.array(dtype=wp.float32, ndim=3),
        depth_out: wp.array(dtype=wp.uint16, ndim=2),
    ):
        i, j = wp.tid()
        val = depth_in[i, j, 0] * 65535.0
        if val < 0.0:
            val = 0.0
        if val > 65535.0:
            val = 65535.0
        depth_out[i, j] = wp.uint16(val)

    @staticmethod
    @wp.kernel
    def _depth_f16_to_uint16_kernel(
        depth_in: wp.array(dtype=wp.float16, ndim=3),
        depth_out: wp.array(dtype=wp.uint16, ndim=2),
    ):
        i, j = wp.tid()
        val = wp.float32(depth_in[i, j, 0]) * 65535.0
        if val < 0.0:
            val = 0.0
        if val > 65535.0:
            val = 65535.0
        depth_out[i, j] = wp.uint16(val)

    # ---- Copy-to-slot kernels (extract ch-0 into f32 buffer) -----------------

    @staticmethod
    @wp.kernel
    def _copy_extract_f32_kernel(
        src: wp.array(dtype=wp.float32, ndim=3),    # (H, W, C)
        dst: wp.array(dtype=wp.float32, ndim=3),    # (N, H, W)
        slot: int,
    ):
        """Extract channel 0 from (H,W,C) float32 -> (N,H,W) float32 slot."""
        i, j = wp.tid()
        dst[slot, i, j] = src[i, j, 0]

    @staticmethod
    @wp.kernel
    def _copy_extract_f16_to_f32_kernel(
        src: wp.array(dtype=wp.float16, ndim=3),    # (H, W, C)
        dst: wp.array(dtype=wp.float32, ndim=3),    # (N, H, W)
        slot: int,
    ):
        """Extract channel 0 from (H,W,C) float16 -> (N,H,W) float32 slot."""
        i, j = wp.tid()
        dst[slot, i, j] = wp.float32(src[i, j, 0])

    # ---- Batch conversion kernel (all N frames at once) ----------------------

    @staticmethod
    @wp.kernel
    def _batch_convert_kernel(
        raw_buf: wp.array(dtype=wp.float32, ndim=3),  # (N, H, W)
        out_buf: wp.array(dtype=wp.uint16, ndim=3),    # (N, H, W)
    ):
        n, i, j = wp.tid()
        val = raw_buf[n, i, j] * 65535.0
        if val < 0.0:
            val = 0.0
        if val > 65535.0:
            val = 65535.0
        out_buf[n, i, j] = wp.uint16(val)

    # ---- Single-frame visualisation (also used by aov_capture_all.py) --------

    @staticmethod
    def visualize_depth(depth, out_path, device=None, verbose=False):
        """Convert depth to 16-bit PNG and write to disk.

        Accepts either a ``wp.array`` on CUDA (processed via Warp kernel)
        or a ``np.ndarray`` (CPU fallback).
        """
        t0 = time.perf_counter()

        if isinstance(depth, wp.array) and "cuda" in str(depth.device):
            dev = str(depth.device)
            h, w = depth.shape[0], depth.shape[1]
            out_wp = wp.zeros((h, w), dtype=wp.uint16, device=dev)

            if depth.ndim == 3:
                kernel = (DepthCapture._depth_f32_to_uint16_kernel
                          if depth.dtype == wp.float32
                          else DepthCapture._depth_f16_to_uint16_kernel)
                wp.launch(
                    kernel=kernel,
                    dim=(h, w), inputs=[depth, out_wp], device=dev,
                )
            elif depth.ndim == 2:
                # 2D array - fall back to numpy (rare path)
                depth_np = depth.numpy()
                depth_16bit = (depth_np * 65535).astype(np.uint16)
                cv2.imwrite(out_path, depth_16bit)
                return
            else:
                depth_np = depth.numpy()
                depth_np = (depth_np[:, :, 0] if depth_np.ndim == 3
                            else depth_np)
                depth_16bit = (depth_np * 65535).astype(np.uint16)
                cv2.imwrite(out_path, depth_16bit)
                return

            depth_16bit = out_wp.numpy()
        else:
            depth = (depth if isinstance(depth, np.ndarray)
                     else np.array(depth))
            depth = depth[:, :, 0] if depth.ndim == 3 else depth
            depth_16bit = (depth * 65535).astype(np.uint16)

        cv2.imwrite(out_path, depth_16bit)

        if verbose:
            t_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [depth] {t_ms:.2f} ms  {out_path}")

    # ---- Capture loop --------------------------------------------------------

    async def capture_and_visualize(self):
        render_product = rep.create.render_product(
            self.camera_prim_path, self.resolution)
        depth_annot = rep.AnnotatorRegistry.get_annotator("PtZDepth")
        depth_annot.attach(render_product)

        start_tc = int(round(self.start_time * self.tcps))
        print(f"[DepthCapture] Starting - {self.num_timeline_frames} frames, "
              f"write_interval={self.write_interval}, "
              f"format={self.output_format}")

        t_total_start = time.perf_counter()

        # Batch buffer state - lazy-allocated on first frame
        self._raw_buf = None
        self._out_buf = None
        batch_paths = []
        slot = 0

        for timecode in range(start_tc, start_tc + self.num_timeline_frames):
            # ---- Set timeline to current timecode and render ------------------
            self.timeline.set_current_time(timecode / self.tcps)
            await omni.kit.app.get_app().next_update_async()
            await rep.orchestrator.step_async()

            ext = "png" if self.output_format == "png" else "npz"
            out_path = os.path.join(
                self.output_dir, f'Capture.{timecode:04d}.{ext}')

            # ---- Phase 1: Collect - get data + copy raw into slot (async) ----
            depth_cuda = depth_annot.get_data(device=self.device)
            h, w = depth_cuda.shape[0], depth_cuda.shape[1]
            self._ensure_buffers(h, w)

            self._copy_to_raw_slot(depth_cuda, slot)
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

        depth_annot.detach(render_product)
        render_product.destroy()

        print(f"\n[DepthCapture] Done - {self.num_timeline_frames} frames in "
              f"{t_total:.2f} s  (avg {avg_ms:.2f} ms/frame)")

    # ---- Internal helpers ----------------------------------------------------

    def _ensure_buffers(self, h, w):
        """Lazy-allocate raw input buffer (N,H,W) f32 and uint16 output buffer."""
        N = self.write_interval
        if self._out_buf is not None and self._out_buf.shape[1:] == (h, w):
            return
        self._out_buf = wp.zeros((N, h, w), dtype=wp.uint16,
                                 device=self.device)
        self._raw_buf = wp.zeros((N, h, w), dtype=wp.float32,
                                 device=self.device)

    def _copy_to_raw_slot(self, depth_cuda, slot):
        """Extract depth channel 0 into ``_raw_buf[slot]`` (async).

        Handles both float32 and float16 annotator outputs.
        """
        h, w = depth_cuda.shape[0], depth_cuda.shape[1]
        dev = str(depth_cuda.device)
        kernel = (DepthCapture._copy_extract_f32_kernel
                  if depth_cuda.dtype == wp.float32
                  else DepthCapture._copy_extract_f16_to_f32_kernel)
        wp.launch(
            kernel=kernel,
            dim=(h, w),
            inputs=[depth_cuda, self._raw_buf, slot],
            device=dev,
        )

    def _batch_convert(self, count):
        """Convert ``count`` raw frames -> uint16 greyscale in one kernel."""
        h, w = self._out_buf.shape[1], self._out_buf.shape[2]
        wp.launch(
            kernel=DepthCapture._batch_convert_kernel,
            dim=(count, h, w),
            inputs=[self._raw_buf, self._out_buf],
            device=self.device,
        )

    def _flush_buffer(self, paths, count):
        """Batch-convert -> single .numpy() -> write (PNG parallel or NPZ)."""
        if count == 0:
            return
        t0 = time.perf_counter()

        # 1. Convert all raw frames -> uint16 greyscale (one kernel launch)
        self._batch_convert(count)

        # 2. One GPU->CPU transfer for the entire batch
        batch_np = self._out_buf.numpy()

        # 3. Write frames to disk
        if self.output_format == "png":
            # Parallel cv2.imwrite - PNG compression releases the GIL
            with ThreadPoolExecutor(max_workers=count) as pool:
                futures = [
                    pool.submit(cv2.imwrite, paths[i], batch_np[i])
                    for i in range(count)
                ]
                for f in futures:
                    f.result()      # propagate any exceptions
        else:
            # NPZ - raw numpy save (no compression, fastest)
            for i in range(count):
                np.save(paths[i], batch_np[i])

        if self.verbose:
            t_ms = (time.perf_counter() - t0) * 1000.0
            first = os.path.basename(paths[0])
            last = os.path.basename(paths[count - 1])
            fmt = "PNG parallel" if self.output_format == "png" else "NPZ"
            print(f"  [depth] Flushed {count} frame(s): "
                  f"{first} -> {last}  ({t_ms:.2f} ms, {fmt})")


if __name__ == "__main__":
    camera_prim_path = "/World/CapsuleCam/Camera"
    output_dir = ""
    depth_capturer = DepthCapture(
        camera_prim_path=camera_prim_path,
        output_dir=output_dir,
        resolution=(512, 512),
        verbose=True,
        write_interval=5,
    )
    asyncio.ensure_future(depth_capturer.capture_and_visualize())
