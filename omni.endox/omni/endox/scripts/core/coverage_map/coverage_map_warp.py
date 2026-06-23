import os
import warp as wp
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf
import omni.usd
import omni.timeline
import gc
import time
import asyncio

EPS = 1e-8
WARP_EPS = wp.constant(EPS)


# --- Utilities ---
def get_view_projection(camera_prim):
    camera_geom = UsdGeom.Camera(camera_prim)
    cam = camera_geom.GetCamera()
    frustum = cam.frustum
    view_matrix = np.array(frustum.ComputeViewMatrix())
    proj_matrix = np.array(frustum.ComputeProjectionMatrix())
    return view_matrix, proj_matrix

def get_world_transform(prim):
    mat = omni.usd.get_world_transform_matrix(prim)
    return np.array(mat).reshape((4, 4))

def ndc_to_pixel_host(x, y, resolution_x, resolution_y):
    px = int((x + 1.0) * 0.5 * (resolution_x - 1))
    py = int((1.0 - y) * 0.5 * (resolution_y - 1))
    return px, py

def precompute_tri_indices(faces_np, counts_np):
    tri_indices = []
    face_idx = 0
    for count in counts_np:
        for i in range(1, count - 1):
            tri_indices.append([
                faces_np[face_idx],
                faces_np[face_idx + i],
                faces_np[face_idx + i + 1]
            ])
        face_idx += count
    return np.array(tri_indices, dtype=np.int32)

# --- Warp functions/kernels ---
@wp.func
def ndc_to_pixel(x: float, y: float, resolution_x: int, resolution_y: int):
    px = wp.int((x + 1.0) * 0.5 * float(resolution_x - 1))
    py = wp.int((1.0 - y) * 0.5 * float(resolution_y - 1))
    return px, py

@wp.func
def area(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

@wp.func
def barycentric(pt: wp.vec2f, v0: wp.vec2f, v1: wp.vec2f, v2: wp.vec2f):
    area_total = area(v0, v1, v2)
    if wp.abs(area_total) < WARP_EPS:
        return 0.0, 0.0, 0.0
    w0 = area(pt, v1, v2) / area_total
    w1 = area(pt, v2, v0) / area_total
    w2 = area(pt, v0, v1) / area_total
    return w0, w1, w2

@wp.func
def project_vertex(
    pt_local: wp.vec3f,
    world_transform: wp.mat44f,
    view_matrix: wp.mat44f,
    proj_matrix: wp.mat44f
):
    pt4 = wp.vec4f(pt_local[0], pt_local[1], pt_local[2], 1.0) @ world_transform 
    view_pt = pt4 @ view_matrix 
    clip_pt = view_pt @ proj_matrix 
    w = clip_pt[3]
    failed = (w <= 0.0)
    ndc = wp.vec3f(0.0, 0.0, 0.0)
    visible = False
    if not failed:
        ndc = wp.vec3f(clip_pt[0]/w, clip_pt[1]/w, clip_pt[2]/w)
        visible = (-1.0 <= ndc[0] <= 1.0) and (-1.0 <= ndc[1] <= 1.0) and (0.0 < ndc[2] <= 1.0)
    return visible, ndc

@wp.kernel
def project_vertices_kernel(
    pts_local: wp.array(dtype=wp.vec3f),
    world_transform: wp.mat44f,
    view_matrix: wp.mat44f,
    proj_matrix: wp.mat44f,
    visible: wp.array(dtype=wp.int32),
    ndc: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    vis, ndc_pt = project_vertex(pts_local[i], world_transform, view_matrix, proj_matrix)
    visible[i] = int(vis)
    ndc[i] = ndc_pt

@wp.kernel
def rasterize_mesh_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),   # [num_triangles, 3]
    ndc: wp.array(dtype=wp.vec3f),           # all vertex ndcs
    visible: wp.array(dtype=wp.int32),       # all vertex visibilities
    resolution_x: int,
    resolution_y: int,
    z_buffer: wp.array2d(dtype=wp.float32)
):
    tid = wp.tid()
    idx0 = tri_indices[tid, 0]
    idx1 = tri_indices[tid, 1]
    idx2 = tri_indices[tid, 2]
    vis0 = visible[idx0]
    vis1 = visible[idx1]
    vis2 = visible[idx2]
    if (vis0 == 0 and vis1 == 0 and vis2 == 0):
        return
    ndc0 = ndc[idx0]
    ndc1 = ndc[idx1]
    ndc2 = ndc[idx2]
    px0, py0 = ndc_to_pixel(ndc0[0], ndc0[1], resolution_x, resolution_y)
    px1, py1 = ndc_to_pixel(ndc1[0], ndc1[1], resolution_x, resolution_y)
    px2, py2 = ndc_to_pixel(ndc2[0], ndc2[1], resolution_x, resolution_y)
    area_total = float((px1 - px0) * (py2 - py0) - (py1 - py0) * (px2 - px0))
    if wp.abs(area_total) < WARP_EPS:
        return
    xmin = wp.clamp(wp.min(wp.min(px0, px1), px2), 0, resolution_x-1)
    xmax = wp.clamp(wp.max(wp.max(px0, px1), px2), 0, resolution_x-1)
    ymin = wp.clamp(wp.min(wp.min(py0, py1), py2), 0, resolution_y-1)
    ymax = wp.clamp(wp.max(wp.max(py0, py1), py2), 0, resolution_y-1)
    for px in range(xmin, xmax+1):
        for py in range(ymin, ymax+1):
            w0, w1, w2 = barycentric(
                wp.vec2f(float(px), float(py)),
                wp.vec2f(float(px0), float(py0)),
                wp.vec2f(float(px1), float(py1)),
                wp.vec2f(float(px2), float(py2)),
            )
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                z = w0*ndc0[2] + w1*ndc1[2] + w2*ndc2[2]
                wp.atomic_min(z_buffer, px, py, z)


@wp.kernel
def final_visibility_kernel(
    ndc: wp.array(dtype=wp.vec3f),           # all vertex ndcs
    visible: wp.array(dtype=wp.int32),       # all vertex visibilities
    resolution_x: int,
    resolution_y: int,
    z_buffer: wp.array2d(dtype=wp.float32),
    visibles: wp.array(dtype=wp.int32),      # output per-vertex final visibility
):
    v = wp.tid()
    ndc_pt = ndc[v]
    px, py = ndc_to_pixel(ndc_pt[0], ndc_pt[1], resolution_x, resolution_y)
    if visible[v] != 0 and 0 <= px and px < resolution_x and 0 <= py and py < resolution_y:
        zbuf = z_buffer[px, py]
        if wp.abs(ndc_pt[2] - zbuf) < WARP_EPS and ndc_pt[2] <= zbuf + WARP_EPS:
            visibles[v] = 1
        else:
            visibles[v] = 0
    else:
        visibles[v] = 0


# --- Main coverage function, using warp ---
def compute_coverage_map_at_timecode(mesh_prim, camera_prim, points_np, faces_np, counts_np, tri_indices_wp, device,resolution=512):
    # Load transform/camera/mesh data
    view_matrix, proj_matrix = get_view_projection(camera_prim)
    world_transform_np = get_world_transform(mesh_prim)

    resolution_x = resolution_y = resolution
    num_vertices = points_np.shape[0]

    # Warp arrays  
    pts_local = wp.array(points_np, dtype=wp.vec3f, device=device)
    world_transform = wp.mat44f(*world_transform_np.flatten())
    view_matrix_w = wp.mat44f(*view_matrix.flatten())
    proj_matrix_w = wp.mat44f(*proj_matrix.flatten())
    visible = wp.zeros(num_vertices, dtype=wp.int32, device=device)
    ndc = wp.zeros(num_vertices, dtype=wp.vec3f, device=device)

    # Warp: project all vertices
    wp.launch(
        project_vertices_kernel,
        dim=num_vertices,
        inputs=[pts_local, world_transform, view_matrix_w, proj_matrix_w, visible, ndc],
        device=device
    )

    num_triangles = tri_indices_wp.shape[0]
    z_buffer = wp.array(np.full((resolution_x, resolution_y), np.inf, dtype=np.float32), dtype=wp.float32, device=device)
    
    wp.launch(
        rasterize_mesh_kernel,
        dim=num_triangles,
        inputs=[tri_indices_wp, ndc, visible, resolution_x, resolution_y, z_buffer],
        device=device
    )

    # Warp: final visibility per-vertex (z-buffer test)
    visibles_wp = wp.zeros(num_vertices, dtype=wp.int32, device=device)
    wp.launch(
        final_visibility_kernel,
        dim=num_vertices,
        inputs=[ndc, visible, resolution_x, resolution_y, z_buffer, visibles_wp],
        device=device
    )
    visibles = visibles_wp.numpy()

    print(f"{int(np.sum(visibles == 1))} visible points colored red.")
    print("Mesh colored by camera coverage.")
    gc.collect()
    return visibles


# ---------------------------------------------------------------------------
# CoverageMapComputer - interactive class with pause / stop / resume
# ---------------------------------------------------------------------------
class CoverageMapComputer:
    """Computes per-vertex camera visibility across timeline frames.

    Supports pause / resume / stop so the UI can control the long-running
    computation interactively.
    """

    def __init__(
        self,
        mesh_prim,
        camera_prim,
        resolution=512,
        save_path="C:/temp/coverage_vertices.npz",
        per_frame_save_path="C:/temp/coverage_vertices_per_frame.npz",
        num_frames=None,
    ):
        self.mesh_prim = mesh_prim
        self.camera_prim = camera_prim
        self.resolution = resolution
        self.save_path = save_path
        self.per_frame_save_path = per_frame_save_path
        self.num_frames = num_frames

        self._paused = False
        self._stopped = False

    # - Controls ----------------------------------------------------------

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stopped = True
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    # - Main computation --------------------------------------------------

    async def compute(self):
        device = f'cuda:{len(wp.get_cuda_devices()) - 1}'

        timeline = omni.timeline.get_timeline_interface()
        tcps = timeline.get_time_codes_per_seconds()
        start_time = timeline.get_start_time()
        end_time = timeline.get_end_time()
        num_timeline_frames = int((end_time - start_time) * tcps)

        start_tc = int(round(start_time * tcps))

        frames_to_capture = (
            min(self.num_frames, num_timeline_frames)
            if self.num_frames is not None
            else num_timeline_frames
        )
        if frames_to_capture <= 0:
            print("[Coverage Map] No frames to capture "
                  "(empty timeline or num_frames=0).")
            return np.zeros(0, dtype=bool)

        mesh = UsdGeom.Mesh(self.mesh_prim)
        points_np = np.array(
            mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces_np = np.array(
            mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        counts_np = np.array(
            mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        num_vertices = points_np.shape[0]
        tri_indices_np = precompute_tri_indices(faces_np, counts_np)
        tri_indices_wp = wp.array(
            tri_indices_np, dtype=wp.int32, device=device)

        coverage = np.zeros(num_vertices, dtype=bool)
        per_frame_coverage = np.zeros(
            (frames_to_capture, num_vertices), dtype=np.int32)

        print(f"[Coverage Map] Capturing {frames_to_capture} frames "
              f"(timeline has {num_timeline_frames}).")
        captured_frames = 0
        for timecode in range(start_tc, start_tc + frames_to_capture):
            while self._paused and not self._stopped:
                await asyncio.sleep(0.1)
            if self._stopped:
                print(f"[Coverage Map] Stopped at frame {timecode}")
                break

            timeline.set_current_time(timecode / tcps)
            frame_index = timecode - start_tc
            print(f"[Coverage Map] Frame {frame_index} "
                  f"(timecode {timecode})")
            await omni.kit.app.get_app().next_update_async()
            visibles = compute_coverage_map_at_timecode(
                self.mesh_prim, self.camera_prim, points_np,
                faces_np, counts_np, tri_indices_wp, device)
            coverage |= (visibles > 0)
            per_frame_coverage[frame_index, :] = (
                (visibles > 0).astype(np.int32))
            captured_frames += 1

        if captured_frames < frames_to_capture:
            per_frame_coverage = per_frame_coverage[:captured_frames]

        colors = [Gf.Vec3f(1, 0, 0) if flag else Gf.Vec3f(0, 0, 1)
                  for flag in coverage]

        stage = self.mesh_prim.GetStage()
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
        displayColor = mesh.CreateDisplayColorPrimvar(
            UsdGeom.Tokens.vertex)
        displayColor.Set(colors)

        # RTX ignores displayColor while a material is bound; block the
        # binding on the session layer so the renderer falls back to vertex
        # displayColor (reversible).
        binding_api = UsdShade.MaterialBindingAPI.Apply(self.mesh_prim)
        binding_api.UnbindAllBindings()

        await omni.kit.app.get_app().next_update_async()

        red_vertex_flags = coverage.astype(np.int32)
        red_vertex_indices = np.nonzero(
            red_vertex_flags)[0].astype(np.int32)

        for path in (self.save_path, self.per_frame_save_path):
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)

        np.savez(
            self.save_path,
            red_flags=red_vertex_flags,
            red_indices=red_vertex_indices,
        )
        print(f"Saved red-vertex flags and indices to: {self.save_path}")

        np.savez(
            self.per_frame_save_path,
            per_frame_flags=per_frame_coverage,
        )
        print(f"Saved per-frame vertex flags to: "
              f"{self.per_frame_save_path}")

        return coverage


async def compute_coverage_map(
    mesh_prim,
    camera_prim,
    resolution=512,
    save_path="C:/temp/coverage_vertices.npz",
    per_frame_save_path="C:/temp/coverage_vertices_per_frame.npz",
    num_frames=None,
):
    """Convenience wrapper - backward-compatible standalone function."""
    computer = CoverageMapComputer(
        mesh_prim, camera_prim,
        resolution=resolution,
        save_path=save_path,
        per_frame_save_path=per_frame_save_path,
        num_frames=num_frames,
    )
    return await computer.compute()


def main():
    start = time.time()

    wp.init()
    stage = omni.usd.get_context().get_stage()
    mesh_prim = stage.GetPrimAtPath("/World/OrganEnv/Colon/Colon")  # Update this path
    camera_prim = stage.GetPrimAtPath("/World/CapsuleCam/Camera")   # Update this path

    asyncio.ensure_future(
        compute_coverage_map(
            mesh_prim,
            camera_prim,
            resolution=512,
            save_path="C:/output/aov_capture/coverage_map/coverage_vertices.npz",
            per_frame_save_path="C:/output/aov_capture/coverage_map/coverage_vertices_per_frame.npz",
            num_frames=None,
         )
    )
    print(f"Time: {time.time() - start}")


if __name__ == "__main__":
    main()
