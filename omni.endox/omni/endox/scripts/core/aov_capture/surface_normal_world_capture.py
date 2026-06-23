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


class WorldSurfaceNormalCapture:
    """Capture world-space surface normals using Warp-accelerated GPU batch pipeline.

    Pipeline (per batch of ``write_interval`` frames):
        1. **Collect** - ``get_data(device=cuda)`` -> extract XYZ channels
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
    def _normals_f32_to_uint16_kernel(
        normals_in: wp.array(dtype=wp.float32, ndim=3),
        normals_out: wp.array(dtype=wp.uint16, ndim=3),
    ):
        i, j = wp.tid()
        nx = normals_in[i, j, 0]
        ny = normals_in[i, j, 1]
        nz = normals_in[i, j, 2]
        if nx < -1.0:
            nx = -1.0
        if nx > 1.0:
            nx = 1.0
        if ny < -1.0:
            ny = -1.0
        if ny > 1.0:
            ny = 1.0
        if nz < -1.0:
            nz = -1.0
        if nz > 1.0:
            nz = 1.0
        normals_out[i, j, 0] = wp.uint16((nz + 1.0) / 2.0 * 65535.0)
        normals_out[i, j, 1] = wp.uint16((ny + 1.0) / 2.0 * 65535.0)
        normals_out[i, j, 2] = wp.uint16((nx + 1.0) / 2.0 * 65535.0)

    # - Single-frame: f16 input -> uint16 BGR --
    @staticmethod
    @wp.kernel
    def _normals_f16_to_uint16_kernel(
        normals_in: wp.array(dtype=wp.float16, ndim=3),
        normals_out: wp.array(dtype=wp.uint16, ndim=3),
    ):
        i, j = wp.tid()
        nx = wp.float32(normals_in[i, j, 0])
        ny = wp.float32(normals_in[i, j, 1])
        nz = wp.float32(normals_in[i, j, 2])
        if nx < -1.0:
            nx = -1.0
        if nx > 1.0:
            nx = 1.0
        if ny < -1.0:
            ny = -1.0
        if ny > 1.0:
            ny = 1.0
        if nz < -1.0:
            nz = -1.0
        if nz > 1.0:
            nz = 1.0
        normals_out[i, j, 0] = wp.uint16((nz + 1.0) / 2.0 * 65535.0)
        normals_out[i, j, 1] = wp.uint16((ny + 1.0) / 2.0 * 65535.0)
        normals_out[i, j, 2] = wp.uint16((nx + 1.0) / 2.0 * 65535.0)

    # ---- Copy-to-slot kernels (extract XYZ into f32 buffer) ------------------

    @staticmethod
    @wp.kernel
    def _copy_extract_normals_f32_kernel(
        src: wp.array(dtype=wp.float32, ndim=3),
        dst: wp.array(dtype=wp.float32, ndim=4),
        slot: int,
    ):
        i, j = wp.tid()
        dst[slot, i, j, 0] = src[i, j, 0]
        dst[slot, i, j, 1] = src[i, j, 1]
        dst[slot, i, j, 2] = src[i, j, 2]

    @staticmethod
    @wp.kernel
    def _copy_extract_normals_f16_to_f32_kernel(
        src: wp.array(dtype=wp.float16, ndim=3),
        dst: wp.array(dtype=wp.float32, ndim=4),
        slot: int,
    ):
        i, j = wp.tid()
        dst[slot, i, j, 0] = wp.float32(src[i, j, 0])
        dst[slot, i, j, 1] = wp.float32(src[i, j, 1])
        dst[slot, i, j, 2] = wp.float32(src[i, j, 2])

    # ---- Batch conversion kernel (f32 -> uint16 BGR) -------------------------

    @staticmethod
    @wp.kernel
    def _batch_convert_normals_kernel(
        raw_buf: wp.array(dtype=wp.float32, ndim=4),
        out_buf: wp.array(dtype=wp.uint16, ndim=4),
    ):
        n, i, j = wp.tid()
        nx = raw_buf[n, i, j, 0]
        ny = raw_buf[n, i, j, 1]
        nz = raw_buf[n, i, j, 2]
        if nx < -1.0:
            nx = -1.0
        if nx > 1.0:
            nx = 1.0
        if ny < -1.0:
            ny = -1.0
        if ny > 1.0:
            ny = 1.0
        if nz < -1.0:
            nz = -1.0
        if nz > 1.0:
            nz = 1.0
        out_buf[n, i, j, 0] = wp.uint16((nz + 1.0) / 2.0 * 65535.0)
        out_buf[n, i, j, 1] = wp.uint16((ny + 1.0) / 2.0 * 65535.0)
        out_buf[n, i, j, 2] = wp.uint16((nx + 1.0) / 2.0 * 65535.0)

    # ---- Single-frame visualisation (also used by aov_capture_all.py) --------

    @staticmethod
    def visualize_normals(normals, out_path, verbose=False):
        """Convert normals to 16-bit BGR PNG and write to disk.

        Accepts ``wp.array`` on CUDA (float16 or float32) or ``np.ndarray``.
        """
        t0 = time.perf_counter()

        if isinstance(normals, wp.array) and "cuda" in str(normals.device):
            dev = str(normals.device)
            h, w = normals.shape[0], normals.shape[1]

            if normals.ndim != 3:
                normals_np = normals.numpy()
                WorldSurfaceNormalCapture._visualize_normals_numpy(
                    normals_np, out_path)
                return

            out_wp = wp.zeros((h, w, 3), dtype=wp.uint16, device=dev)
            kernel = (WorldSurfaceNormalCapture._normals_f32_to_uint16_kernel
                      if normals.dtype == wp.float32
                      else WorldSurfaceNormalCapture._normals_f16_to_uint16_kernel)
            wp.launch(
                kernel=kernel,
                dim=(h, w), inputs=[normals, out_wp], device=dev,
            )

            bgr_16bit = out_wp.numpy()
        else:
            normals = (normals if isinstance(normals, np.ndarray)
                       else np.array(normals))
            if normals.ndim == 3 and normals.shape[2] > 3:
                normals = normals[:, :, :3]
            x_map = ((np.clip(normals[:, :, 0], -1, 1) + 1) / 2 * 65535).astype(np.uint16)
            y_map = ((np.clip(normals[:, :, 1], -1, 1) + 1) / 2 * 65535).astype(np.uint16)
            z_map = ((np.clip(normals[:, :, 2], -1, 1) + 1) / 2 * 65535).astype(np.uint16)
            h, w = x_map.shape
            bgr_16bit = np.zeros((h, w, 3), dtype=np.uint16)
            bgr_16bit[:, :, 0] = z_map
            bgr_16bit[:, :, 1] = y_map
            bgr_16bit[:, :, 2] = x_map

        cv2.imwrite(out_path, bgr_16bit)

        if verbose:
            t_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [normals] {t_ms:.2f} ms  {out_path}")

    @staticmethod
    def _visualize_normals_numpy(normals_np, out_path):
        if normals_np.ndim == 3 and normals_np.shape[2] > 3:
            normals_np = normals_np[:, :, :3]
        x_map = ((np.clip(normals_np[:, :, 0], -1, 1) + 1) / 2 * 65535).astype(np.uint16)
        y_map = ((np.clip(normals_np[:, :, 1], -1, 1) + 1) / 2 * 65535).astype(np.uint16)
        z_map = ((np.clip(normals_np[:, :, 2], -1, 1) + 1) / 2 * 65535).astype(np.uint16)
        h, w = x_map.shape
        bgr_16bit = np.zeros((h, w, 3), dtype=np.uint16)
        bgr_16bit[:, :, 0] = z_map
        bgr_16bit[:, :, 1] = y_map
        bgr_16bit[:, :, 2] = x_map
        cv2.imwrite(out_path, bgr_16bit)

    # ---- Capture loop --------------------------------------------------------

    async def capture_and_visualize(self):
        render_product = rep.create.render_product(
            self.camera_prim_path, self.resolution)
        normals_annot = rep.AnnotatorRegistry.get_annotator("normals")
        normals_annot.attach(render_product)

        rep.orchestrator.set_capture_on_play(False)

        start_tc = int(round(self.start_time * self.tcps))

        print(f"[WorldNormals] Starting - "
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
            self.timeline.set_current_time(timecode / self.tcps)
            await omni.kit.app.get_app().next_update_async()
            await rep.orchestrator.step_async()

            ext = "png" if self.output_format == "png" else "npz"
            out_path = os.path.join(
                self.output_dir, f'surf_normals_{timecode:04d}.{ext}')

            normals_cuda = normals_annot.get_data(device=self.device)
            h, w = normals_cuda.shape[0], normals_cuda.shape[1]
            self._ensure_buffers(h, w)

            self._copy_to_raw_slot(normals_cuda, slot)
            batch_paths.append(out_path)
            slot += 1

            is_last = (timecode == start_tc + self.num_timeline_frames - 1)
            if slot >= self.write_interval or is_last:
                self._flush_buffer(batch_paths, slot)
                batch_paths.clear()
                slot = 0

        t_total = time.perf_counter() - t_total_start
        avg_ms = (t_total / max(self.num_timeline_frames, 1)) * 1000.0

        normals_annot.detach(render_product)
        render_product.destroy()

        print(f"\n[WorldNormals] Done - "
              f"{self.num_timeline_frames} frames in "
              f"{t_total:.2f} s  (avg {avg_ms:.2f} ms/frame)")

    # ---- Internal helpers ----------------------------------------------------

    def _ensure_buffers(self, h, w):
        N = self.write_interval
        if self._out_buf is not None and self._out_buf.shape[1:3] == (h, w):
            return
        self._out_buf = wp.zeros((N, h, w, 3), dtype=wp.uint16,
                                 device=self.device)
        self._raw_buf = wp.zeros((N, h, w, 3), dtype=wp.float32,
                                 device=self.device)

    def _copy_to_raw_slot(self, normals_cuda, slot):
        h, w = normals_cuda.shape[0], normals_cuda.shape[1]
        dev = str(normals_cuda.device)
        kernel = (WorldSurfaceNormalCapture._copy_extract_normals_f32_kernel
                  if normals_cuda.dtype == wp.float32
                  else WorldSurfaceNormalCapture._copy_extract_normals_f16_to_f32_kernel)
        wp.launch(
            kernel=kernel,
            dim=(h, w),
            inputs=[normals_cuda, self._raw_buf, slot],
            device=dev,
        )

    def _batch_convert(self, count):
        h, w = self._out_buf.shape[1], self._out_buf.shape[2]
        wp.launch(
            kernel=WorldSurfaceNormalCapture._batch_convert_normals_kernel,
            dim=(count, h, w),
            inputs=[self._raw_buf, self._out_buf],
            device=self.device,
        )

    def _flush_buffer(self, paths, count):
        if count == 0:
            return
        t0 = time.perf_counter()

        self._batch_convert(count)
        batch_np = self._out_buf.numpy()

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
            print(f"  [normals] Flushed {count} frame(s): "
                  f"{first} -> {last}  ({t_ms:.2f} ms, {fmt})")


if __name__ == '__main__':
    camera_prim_path = "/World/CapsuleCam/Camera"
    output_dir = "C:/output/aov_capture/surface_normal_world"
    normal_capturer = WorldSurfaceNormalCapture(
        camera_prim_path=camera_prim_path,
        output_dir=output_dir,
        resolution=(512, 512),
        verbose=True,
        write_interval=5,
    )
    asyncio.ensure_future(normal_capturer.capture_and_visualize())
