import omni.replicator.core as rep
import omni.usd
import omni.timeline
import asyncio
import numpy as np
import warp as wp
wp.init()
import _warp_compat  # noqa: F401  (patches wp.types.array for replicator compat)
import carb
from PIL import Image
import os
import time
from concurrent.futures import ThreadPoolExecutor


class OcclusionCapture:
    """GPU-accelerated occlusion mask capture via clipping-plane sweep.

    Pipeline (per timeline frame):
        1. Sweep near-clip across ``num_clipping_steps`` distances, fetching
           each RGB frame directly on CUDA via ``get_data(device="cuda")``.
        2. A Warp kernel converts uint8 RGBA to float32 RGB and writes each
           frame into a pre-allocated GPU buffer slot.
        3. An occlusion kernel compares every frame against the initial,
           accumulating a per-pixel hit count on the GPU.
        4. A threshold kernel converts the cumulative map to uint8 RGBA.

    Across timeline frames, masks are batched into a ``(write_interval, H, W, 4)``
    GPU buffer, transferred once, and written in parallel.
    """

    def __init__(
        self,
        camera_prim_path,
        output_dir,
        num_clipping_steps=100,
        near=1.0,
        far=100.0,
        near_increment=1,
        resolution=(512, 512),
        diff_threshold=0.05,
        black_eps=0.02,
        write_interval=1,
        verbose=False,
        output_format="png",
        num_frames=None,
    ):
        self.camera_prim_path = camera_prim_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.num_clipping_steps = num_clipping_steps
        self.near = near
        self.far = far
        self.near_increment = near_increment
        self.resolution = resolution
        self.diff_threshold = diff_threshold
        self.black_eps = black_eps
        self.write_interval = max(1, write_interval)
        self.verbose = verbose
        self.output_format = output_format.lower()

        self._paused = False
        self._stopped = False
        if self.output_format not in ("png", "npz"):
            raise ValueError(
                f"output_format must be 'png' or 'npz', got '{output_format}'")

        # USD / timeline
        self.stage = omni.usd.get_context().get_stage()
        self.stage_camera = self.stage.GetPrimAtPath(self.camera_prim_path)
        self.clipping_range_attr = self.stage_camera.GetAttribute(
            "clippingRange")
        self.timeline = omni.timeline.get_timeline_interface()
        self.tcps = self.timeline.get_time_codes_per_seconds()
        self.start_time = self.timeline.get_start_time()
        self.end_time = self.timeline.get_end_time()
        tl_frames = int(
            (self.end_time - self.start_time) * self.tcps)
        self.num_timeline_frames = (min(num_frames, tl_frames)
                                    if num_frames else tl_frames)

        # Warp
        wp.init()
        cuda_idx = len(wp.get_cuda_devices()) - 1
        self.device = f"cuda:{cuda_idx}"

        self._clip_buf = None
        self._out_buf = None

    # ---- GPU kernels (accelerated path) ------------------------------------

    @staticmethod
    @wp.kernel
    def _copy_rgb_u8_to_f32_kernel(
        src: wp.array(dtype=wp.uint8, ndim=3),
        dst: wp.array(dtype=wp.float32, ndim=4),
        slot: int,
    ):
        """Copy uint8 RGBA (H,W,4) -> float32 RGB (N,H,W,3) at slot, /255."""
        i, j = wp.tid()
        dst[slot, i, j, 0] = wp.float32(src[i, j, 0]) / 255.0
        dst[slot, i, j, 1] = wp.float32(src[i, j, 1]) / 255.0
        dst[slot, i, j, 2] = wp.float32(src[i, j, 2]) / 255.0

    @staticmethod
    @wp.kernel
    def _copy_rgb_f32_to_f32_kernel(
        src: wp.array(dtype=wp.float32, ndim=3),
        dst: wp.array(dtype=wp.float32, ndim=4),
        slot: int,
    ):
        """Copy float32 RGBA (H,W,4) -> float32 RGB (N,H,W,3) at slot."""
        i, j = wp.tid()
        dst[slot, i, j, 0] = src[i, j, 0]
        dst[slot, i, j, 1] = src[i, j, 1]
        dst[slot, i, j, 2] = src[i, j, 2]

    @staticmethod
    @wp.kernel
    def _occlusion_kernel_f32(
        frames: wp.array(dtype=wp.float32, ndim=4),
        cum_map: wp.array(dtype=wp.int32, ndim=2),
        num_frames: int,
        diff_thresh: float,
        black_eps: float,
    ):
        """Compare each clipping frame against initial; increment if occluded."""
        i, j = wp.tid()
        ir = frames[0, i, j, 0]
        ig = frames[0, i, j, 1]
        ib = frames[0, i, j, 2]
        for k in range(1, num_frames):
            cr = frames[k, i, j, 0]
            cg = frames[k, i, j, 1]
            cb = frames[k, i, j, 2]
            dr = cr - ir
            dg = cg - ig
            db = cb - ib
            dist = wp.sqrt(dr * dr + dg * dg + db * db)
            is_black = ((wp.abs(cr) < black_eps)
                        and (wp.abs(cg) < black_eps)
                        and (wp.abs(cb) < black_eps))
            if dist > diff_thresh and not is_black:
                cum_map[i, j] = cum_map[i, j] + 1

    @staticmethod
    @wp.kernel
    def _threshold_to_rgba_kernel(
        cum_map: wp.array(dtype=wp.int32, ndim=2),
        out: wp.array(dtype=wp.uint8, ndim=4),
        slot: int,
    ):
        """Binary threshold: occluded -> white, visible -> black (RGBA)."""
        i, j = wp.tid()
        val = wp.uint8(0)
        if cum_map[i, j] > 0:
            val = wp.uint8(255)
        out[slot, i, j, 0] = val
        out[slot, i, j, 1] = val
        out[slot, i, j, 2] = val
        out[slot, i, j, 3] = wp.uint8(255)

    # ---- Legacy kernels (backward compat / standalone) ---------------------

    @staticmethod
    @wp.func
    def vec_almost_equal(a: wp.vec3, b: wp.vec3, eps: float):
        return ((wp.abs(a[0] - b[0]) < eps)
                and (wp.abs(a[1] - b[1]) < eps)
                and (wp.abs(a[2] - b[2]) < eps))

    @staticmethod
    @wp.kernel
    def occlusion_kernel(
        frames: wp.array(dtype=wp.vec3, ndim=3),
        initial: wp.array(dtype=wp.vec3, ndim=2),
        cum_map: wp.array(dtype=int, ndim=2),
        num_frames: int,
        black_rgb: wp.vec3,
        diff_thresh: float,
        black_eps: float,
    ):
        i, j = wp.tid()
        pixel_init = initial[i, j]
        for k in range(1, num_frames):
            pixel_curr = frames[k, i, j]
            if ((wp.length(pixel_curr - pixel_init) > diff_thresh)
                    and (not OcclusionCapture.vec_almost_equal(
                        pixel_curr, black_rgb, black_eps))):
                cum_map[i, j] += 1

    @staticmethod
    def capture_occlusion(rgb_frames, device, diff_threshold=0.05,
                          black_eps=0.02):
        """Legacy CPU-upload path. Accepts list of numpy RGB frames."""
        rgb_frames_np = np.array(rgb_frames)
        num_frames, height, width, _ = rgb_frames_np.shape
        rgb_float = rgb_frames_np[..., :3].astype(np.float32) / 255.0

        frames_wp = wp.array(rgb_float, dtype=wp.vec3, device=device)
        initial_wp = wp.array(rgb_float[0], dtype=wp.vec3, device=device)
        cum_map = wp.zeros((height, width), dtype=wp.int32, device=device)

        wp.launch(
            kernel=OcclusionCapture.occlusion_kernel,
            dim=(height, width),
            inputs=[frames_wp, initial_wp, cum_map, num_frames,
                    wp.vec3(0.0, 0.0, 0.0), diff_threshold, black_eps],
            device=device)

        cm = cum_map.numpy()
        result = np.zeros((height, width, 4), dtype=np.uint8)
        result[cm > 0] = [255, 255, 255, 255]
        result[cm == 0] = [0, 0, 0, 255]
        return result

    # ---- Buffer management -------------------------------------------------

    def _ensure_buffers(self, h, w):
        if self._clip_buf is not None:
            return
        N_clip = self.num_clipping_steps
        N_batch = self.write_interval
        self._clip_buf = wp.zeros(
            (N_clip, h, w, 3), dtype=wp.float32, device=self.device)
        self._out_buf = wp.zeros(
            (N_batch, h, w, 4), dtype=wp.uint8, device=self.device)

    # ---- Helpers -----------------------------------------------------------

    @staticmethod
    def _pil_save_rgba(data, path):
        Image.fromarray(data, "RGBA").save(path)

    # ---- Warmup (compile kernels) ------------------------------------------

    def _warmup(self):
        dev = self.device
        u8 = wp.zeros((1, 1, 4), dtype=wp.uint8, device=dev)
        f32 = wp.zeros((1, 1, 4), dtype=wp.float32, device=dev)
        cb = wp.zeros((1, 1, 1, 3), dtype=wp.float32, device=dev)
        cm = wp.zeros((1, 1), dtype=wp.int32, device=dev)
        ob = wp.zeros((1, 1, 1, 4), dtype=wp.uint8, device=dev)

        wp.launch(OcclusionCapture._copy_rgb_u8_to_f32_kernel,
                  dim=(1, 1), inputs=[u8, cb, 0], device=dev)
        wp.launch(OcclusionCapture._copy_rgb_f32_to_f32_kernel,
                  dim=(1, 1), inputs=[f32, cb, 0], device=dev)
        wp.launch(OcclusionCapture._occlusion_kernel_f32,
                  dim=(1, 1), inputs=[cb, cm, 1, 0.05, 0.02], device=dev)
        wp.launch(OcclusionCapture._threshold_to_rgba_kernel,
                  dim=(1, 1), inputs=[cm, ob, 0], device=dev)
        wp.synchronize_device(dev)

    # ---- Pause / Stop controls ------------------------------------------------

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stopped = True
        self._paused = False

    # ---- Main capture loop (GPU-accelerated) --------------------------------

    async def capture_and_visualize(self):
        render_product = rep.create.render_product(
            self.camera_prim_path, self.resolution)
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach(render_product)

        rep.orchestrator.set_capture_on_play(False)
        start_tc = int(round(self.start_time * self.tcps))

        print(f"[Occlusion] {self.num_timeline_frames} frames, "
              f"{self.num_clipping_steps} clipping steps/frame, "
              f"write_interval={self.write_interval}")

        # Save original clipping range so we can restore it on exit
        original_clipping = self.clipping_range_attr.Get()

        # Switch to RTX Real-Time for faster clipping sweeps
        settings = carb.settings.get_settings()
        original_mode = settings.get("/rtx/rendermode")
        if original_mode != "RaytracedLighting":
            print(f"[Occlusion] Switching renderer: {original_mode} "
                  f"-> RaytracedLighting")
            settings.set("/rtx/rendermode", "RaytracedLighting")
            await omni.kit.app.get_app().next_update_async()

        self._warmup()
        t_total = time.perf_counter()

        batch_paths = []
        slot = 0

        captured_frames = 0
        try:
            for timecode in range(start_tc,
                                  start_tc + self.num_timeline_frames):
                if self._paused and not self._stopped:
                    self.clipping_range_attr.Set(original_clipping)
                    while self._paused and not self._stopped:
                        await asyncio.sleep(0.1)
                if self._stopped:
                    if self.verbose:
                        print(f"[Occlusion] Stopped at frame {timecode}")
                    break

                self.timeline.set_current_time(timecode / self.tcps)
                await omni.kit.app.get_app().next_update_async()

                # --- Clipping sweep: collect RGB frames on GPU ---
                for step in range(self.num_clipping_steps):
                    new_near = self.near + step * self.near_increment
                    self.clipping_range_attr.Set((new_near, self.far))
                    await rep.orchestrator.step_async(delta_time=0.0)

                    rgb_cuda = rgb_annot.get_data(device=self.device)
                    h, w = rgb_cuda.shape[0], rgb_cuda.shape[1]
                    self._ensure_buffers(h, w)

                    if rgb_cuda.dtype == wp.uint8:
                        wp.launch(
                            OcclusionCapture._copy_rgb_u8_to_f32_kernel,
                            dim=(h, w),
                            inputs=[rgb_cuda, self._clip_buf, step],
                            device=self.device)
                    else:
                        wp.launch(
                            OcclusionCapture._copy_rgb_f32_to_f32_kernel,
                            dim=(h, w),
                            inputs=[rgb_cuda, self._clip_buf, step],
                            device=self.device)

                # --- Occlusion detection on GPU ---
                cum_map = wp.zeros((h, w), dtype=wp.int32,
                                   device=self.device)
                wp.launch(
                    OcclusionCapture._occlusion_kernel_f32,
                    dim=(h, w),
                    inputs=[self._clip_buf, cum_map,
                            self.num_clipping_steps,
                            self.diff_threshold, self.black_eps],
                    device=self.device)

                # --- Threshold to RGBA on GPU ---
                wp.launch(
                    OcclusionCapture._threshold_to_rgba_kernel,
                    dim=(h, w),
                    inputs=[cum_map, self._out_buf, slot],
                    device=self.device)

                ext = "png" if self.output_format == "png" else "npz"
                out_path = os.path.join(
                    self.output_dir, f"occlusion_{timecode:04d}.{ext}")
                batch_paths.append(out_path)
                slot += 1
                captured_frames += 1

                if self.verbose:
                    print(f"  Frame {timecode:04d}: clipping sweep done")

                # --- Flush batch ---
                is_last = (timecode ==
                           start_tc + self.num_timeline_frames - 1)
                if slot >= self.write_interval or is_last or self._stopped:
                    batch_np = self._out_buf.numpy()

                    if self.output_format == "png":
                        with ThreadPoolExecutor(
                                max_workers=min(slot, 8)) as pool:
                            futs = [
                                pool.submit(self._pil_save_rgba,
                                            batch_np[i], batch_paths[i])
                                for i in range(slot)
                            ]
                            for f in futs:
                                f.result()
                    else:
                        for i in range(slot):
                            np.save(batch_paths[i], batch_np[i])

                    if self.verbose:
                        for p in batch_paths:
                            print(f"    Saved {p}")

                    batch_paths.clear()
                    slot = 0
        finally:
            rgb_annot.detach(render_product)
            render_product.destroy()

            # Restore original camera clipping range
            if original_clipping is not None:
                self.clipping_range_attr.Set(original_clipping)

            # Restore original render mode
            if original_mode and original_mode != "RaytracedLighting":
                print(f"[Occlusion] Restoring renderer: "
                      f"RaytracedLighting -> {original_mode}")
                settings.set("/rtx/rendermode", original_mode)

        elapsed = time.perf_counter() - t_total
        avg_ms = elapsed / max(captured_frames, 1) * 1000.0
        status = "stopped" if self._stopped else "done"
        print(f"\n[Occlusion] {status.capitalize()} - "
              f"{captured_frames}/{self.num_timeline_frames} frames "
              f"in {elapsed:.2f} s  (avg {avg_ms:.2f} ms/frame)")


if __name__ == "__main__":
    camera_prim_path = "/World/CapsuleCam/Camera"
    output_dir = "C:/output/aov_capture/occlusion"
    capturer = OcclusionCapture(
        camera_prim_path=camera_prim_path,
        output_dir=output_dir,
        num_clipping_steps=100,
        near=1.0,
        far=100.0,
        near_increment=1,
        write_interval=5,
        verbose=True,
    )
    asyncio.ensure_future(capturer.capture_and_visualize())
