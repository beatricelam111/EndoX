import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
import omni.usd


def apply_vertex_colors_from_npz(npz_path: str, prim_path: str):
    """Apply coverage vertex colours and disable materials so they are visible.

    Red = visible to camera, Blue = not visible.

    Returns the number of red (visible) and blue (not visible) vertices
    as a ``(red, blue)`` tuple, or ``None`` on failure.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"Prim not found or invalid: {prim_path}")
        return None

    mesh = UsdGeom.Mesh(prim)

    data = np.load(npz_path)

    if "red_flags" in data.files:
        red_flags = data["red_flags"].astype(bool)
        num_vertices = len(red_flags)
        print(f"Loaded red_flags with {num_vertices} entries.")
    elif "red_indices" in data.files:
        red_indices = data["red_indices"].astype(np.int64)
        points_np = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
        num_vertices = points_np.shape[0]
        red_flags = np.zeros(num_vertices, dtype=bool)
        red_flags[red_indices] = True
        print(f"Loaded {len(red_indices)} red vertex indices, "
              f"mesh has {num_vertices} vertices.")
    else:
        print("NPZ does not contain 'red_flags' or 'red_indices'.")
        return None

    points_np = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
    if num_vertices != points_np.shape[0]:
        print(f"Vertex count mismatch: npz={num_vertices}, "
              f"mesh={points_np.shape[0]}")
        return None

    colors = [
        Gf.Vec3f(1.0, 0.0, 0.0) if flag else Gf.Vec3f(0.0, 0.0, 1.0)
        for flag in red_flags
    ]

    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    displayColor = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    displayColor.Set(colors)

    set_materials_visible(prim_path, False)

    n_red = int(red_flags.sum())
    n_blue = num_vertices - n_red
    print(f"Applied vertex colors to {prim_path}: "
          f"{n_red} red (visible), {n_blue} blue (not visible)")
    return n_red, n_blue


# ---------------------------------------------------------------------------
# Material visibility toggle
# ---------------------------------------------------------------------------

# prim_path -> {child_prim_path: material_sdf_path, ...}
_saved_bindings: dict[str, dict[str, "Sdf.Path"]] = {}


def _collect_material_bindings(prim):
    """Return ``{prim_path_str: material_sdf_path}`` for *prim* and all
    descendants that have a direct ``material:binding`` relationship."""
    result = {}
    stack = [prim]
    while stack:
        p = stack.pop()
        api = UsdShade.MaterialBindingAPI(p)
        binding = api.GetDirectBinding()
        mat_path = binding.GetMaterialPath()
        if mat_path and not mat_path.isEmpty:
            result[str(p.GetPath())] = mat_path
        for child in p.GetChildren():
            stack.append(child)
    return result


def set_materials_visible(prim_path: str, visible: bool):
    """Toggle material bindings on *prim_path* and its descendants.

    When *visible* is ``False``, material bindings are saved and then
    unbound so that vertex / displayColor becomes visible.
    When *visible* is ``True``, the saved material bindings are re-applied.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return

    session = stage.GetSessionLayer()
    stage.SetEditTarget(Usd.EditTarget(session))

    if visible:
        saved = _saved_bindings.pop(prim_path, {})
        for child_path, mat_sdf_path in saved.items():
            child_prim = stage.GetPrimAtPath(child_path)
            if not child_prim or not child_prim.IsValid():
                continue
            mat = UsdShade.Material(stage.GetPrimAtPath(mat_sdf_path))
            if not mat:
                continue
            api = UsdShade.MaterialBindingAPI.Apply(child_prim)
            api.Bind(mat)
    else:
        bindings = _collect_material_bindings(prim)
        _saved_bindings[prim_path] = bindings
        for child_path in bindings:
            child_prim = stage.GetPrimAtPath(child_path)
            if not child_prim or not child_prim.IsValid():
                continue
            api = UsdShade.MaterialBindingAPI.Apply(child_prim)
            api.UnbindDirectBinding()


# Example usage for debugging:
def main_debug_recolor():
    npz_path = "C:/output/aov_capture/coverage_map/coverage_vertices.npz"  # path to your saved npz
    prim_path = "/World/OrganEnv/Colon/Colon"   # mesh prim to recolor
    apply_vertex_colors_from_npz(npz_path, prim_path)

if __name__ == "__main__":
    main_debug_recolor()
