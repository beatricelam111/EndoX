"""
EndoX Pipeline - Backend Actions

Thin bridge between the GUI and the bundled EndoX scripts that ship
inside  ``scripts/core/``.  Adds the core sub-directories to
``sys.path`` lazily so that the scripts' own absolute imports
(e.g. ``from depth_capture import DepthCapture``) resolve at runtime.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path resolution - everything is relative to *this* file, so the
# extension remains self-contained regardless of where it is installed.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.join(_SCRIPT_DIR, "core")

_DIRS_TO_ADD = [
    os.path.join(_CORE_DIR, "model_import"),
    os.path.join(_CORE_DIR, "aov_capture"),
    os.path.join(_CORE_DIR, "coverage_map"),
    _CORE_DIR,  # for ``import centreline_extraction``
]

_paths_added = False


def _ensure_paths():
    global _paths_added
    if _paths_added:
        return
    for d in _DIRS_TO_ADD:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    _paths_added = True


# ═══════════════════════════════════════════════════════════════════════════
# 1.  3-D Model Import
# ═══════════════════════════════════════════════════════════════════════════

# ── 1a. DICOM -> Mesh -> Stage ─────────────────────────────────────────────


def download_vista3d_bundle(bundle_dir: str) -> str:
    """Download the Vista3D MONAI bundle into *bundle_dir*.

    Returns the path to the downloaded ``vista3d`` folder.
    Raises ``ImportError`` if monai is not installed.
    """
    import monai.bundle  # noqa: E402

    os.makedirs(bundle_dir, exist_ok=True)
    monai.bundle.download("vista3d", bundle_dir=bundle_dir)
    return os.path.join(bundle_dir, "vista3d")


def _find_kit_python() -> str:
    """Locate the real Python interpreter inside Omniverse Kit.

    ``sys.executable`` in Kit points to ``kit.exe`` which cannot run
    ``python -m ...`` subprocesses.  This searches Kit-internal paths
    and falls back to the system PATH.
    """
    import shutil

    candidates = [
        os.path.join(sys.prefix, "python.exe"),
        os.path.join(sys.prefix, "python"),
        os.path.join(sys.prefix, "bin", "python3"),
        os.path.join(sys.prefix, "bin", "python"),
    ]

    kit_dir = os.path.dirname(sys.executable)
    candidates += [
        os.path.join(kit_dir, "python.exe"),
        os.path.join(kit_dir, "python"),
    ]

    for c in candidates:
        if os.path.isfile(c):
            return c

    found = shutil.which("python3") or shutil.which("python")
    if found:
        return found

    return sys.executable


def run_dicom_to_mesh(
    bundle_path: str,
    dicom_path: str,
    output_obj: str,
    label_id: int = 19,
    upsample: float = 1.0,
    smoothing: float = 0.3,
    sigma: float = 2.0,
    hu_min: float = -500.0,
    hu_max: float = 500.0,
    isovalue: float = 0.0,
    decimation: float = 0.1,
) -> str:
    """Run the DICOM -> NIfTI -> Vista3D -> OBJ pipeline in-process.

    Returns the output OBJ file path on success.
    """
    _ensure_paths()
    import dicom2mesh  # noqa: E402

    python_exe = _find_kit_python()

    obj_path = dicom2mesh.run_pipeline(
        vista3d_path=bundle_path,
        dicom_folder=dicom_path,
        output_obj=output_obj,
        label_id=label_id,
        upsample=upsample,
        smoothing_factor=smoothing,
        gaussian_sigma=sigma,
        hu_min=hu_min,
        hu_max=hu_max,
        isovalue=isovalue,
        decimation_reduction=decimation,
        python_exe=python_exe,
    )
    if obj_path is None or not os.path.isfile(obj_path):
        raise FileNotFoundError(
            f"dicom2mesh did not produce an OBJ file at: {output_obj}"
        )
    return obj_path


# ── 1b. OBJ -> USD -> Stage ────────────────────────────────────────────────


async def import_obj_to_stage(obj_path: str, prim_path: str) -> str:
    """Convert *obj_path* to USD and add it to the current stage.

    Returns the USD file path produced by the converter.
    """
    _ensure_paths()
    from omniverse_import_obj import convert_and_import  # noqa: E402
    return await convert_and_import(obj_path, prim_path)


async def run_dicom_pipeline(
    bundle_path: str,
    dicom_path: str,
    output_obj: str,
    prim_path: str = "/World/DicomMesh",
    label_id: int = 19,
    upsample: float = 1.0,
    smoothing: float = 0.3,
    sigma: float = 2.0,
    hu_min: float = -500.0,
    hu_max: float = 500.0,
    isovalue: float = 0.0,
    decimation: float = 0.1,
) -> str:
    """End-to-end DICOM -> OBJ (in-process) -> USD -> Stage.

    Calls ``dicom2mesh.run_pipeline()`` directly, then converts the
    resulting OBJ to USD and adds it to the current stage.

    Returns the prim path on success, raises on failure.
    """
    run_dicom_to_mesh(
        bundle_path=bundle_path,
        dicom_path=dicom_path,
        output_obj=output_obj,
        label_id=label_id,
        upsample=upsample,
        smoothing=smoothing,
        sigma=sigma,
        hu_min=hu_min,
        hu_max=hu_max,
        isovalue=isovalue,
        decimation=decimation,
    )
    await import_obj_to_stage(output_obj, prim_path)
    return prim_path


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Centreline Extraction
# ═══════════════════════════════════════════════════════════════════════════

def create_centreline_config(
    mesh_prim_path: str = "",
    output_dir: str = "",
    sphere_radius: float = 5.0,
    conda_env: str = "vmtk",
    simplify: bool = True,
    percent_to_keep: float = 0.10,
    curve_type: str = "cubic",
    curve_basis: str = "bspline",
    curve_width: float = 2.0,
):
    """Build and return a ``CentrelineConfig`` populated from GUI values."""
    _ensure_paths()
    from centreline_extraction import CentrelineConfig  # noqa: E402

    cfg = CentrelineConfig()
    cfg.mesh_prim_path = mesh_prim_path
    if output_dir:
        cfg.output_dir = output_dir
    cfg.sphere_radius = sphere_radius
    cfg.conda_env = conda_env
    cfg.simplify = simplify
    cfg.percent_to_keep = percent_to_keep
    cfg.curve_type = curve_type
    cfg.curve_basis = curve_basis
    cfg.curve_width = curve_width
    return cfg


def create_centreline_pipeline(cfg):
    """Create and return a ``CenterlinePipeline`` with the given config."""
    _ensure_paths()
    from centreline_extraction import CenterlinePipeline  # noqa: E402
    return CenterlinePipeline(cfg)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  AOV Capture
# ═══════════════════════════════════════════════════════════════════════════

def create_aov_manager(
    camera_prim_path: str,
    output_dir: str,
    resolution: tuple[int, int] = (512, 512),
    modalities: list[str] | None = None,
    samples_per_pixel: int = 512,
    write_interval: int = 5,
    num_frames: int | None = None,
    num_clipping_steps: int = 100,
    near: float = 1.0,
    far: float = 100.0,
    near_increment: int = 1,
    diff_threshold: float = 0.05,
    black_eps: float = 0.02,
):
    """Create and return an ``AOVCaptureManager``."""
    _ensure_paths()
    from aov_all_capture import AOVCaptureManager  # noqa: E402

    return AOVCaptureManager(
        camera_prim_path=camera_prim_path,
        output_dir=output_dir,
        resolution=resolution,
        modalities=modalities,
        samples_per_pixel=samples_per_pixel,
        verbose=True,
        write_interval=write_interval,
        num_frames=num_frames if num_frames and num_frames > 0 else None,
        num_clipping_steps=num_clipping_steps,
        near=near,
        far=far,
        near_increment=near_increment,
        diff_threshold=diff_threshold,
        black_eps=black_eps,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Coverage Map
# ═══════════════════════════════════════════════════════════════════════════

def create_coverage_map_computer(
    mesh_prim_path: str,
    camera_prim_path: str,
    resolution: int = 512,
    save_path: str = "C:/output/coverage_map/coverage_vertices.npz",
    per_frame_save_path: str = "C:/output/coverage_map/coverage_vertices_per_frame.npz",
    num_frames: int | None = None,
):
    """Create and return a ``CoverageMapComputer`` (supports pause/stop)."""
    _ensure_paths()
    import omni.usd
    from coverage_map_warp import CoverageMapComputer  # noqa: E402

    stage = omni.usd.get_context().get_stage()
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    camera_prim = stage.GetPrimAtPath(camera_prim_path)

    if not mesh_prim.IsValid():
        raise ValueError(f"Mesh prim not found: {mesh_prim_path}")
    if not camera_prim.IsValid():
        raise ValueError(f"Camera prim not found: {camera_prim_path}")

    return CoverageMapComputer(
        mesh_prim,
        camera_prim,
        resolution=resolution,
        save_path=save_path,
        per_frame_save_path=per_frame_save_path,
        num_frames=num_frames if num_frames and num_frames > 0 else None,
    )


async def run_coverage_map(
    mesh_prim_path: str,
    camera_prim_path: str,
    resolution: int = 512,
    save_path: str = "C:/output/coverage_map/coverage_vertices.npz",
    per_frame_save_path: str = "C:/output/coverage_map/coverage_vertices_per_frame.npz",
    num_frames: int | None = None,
):
    """Compute per-vertex camera visibility across timeline frames."""
    computer = create_coverage_map_computer(
        mesh_prim_path=mesh_prim_path,
        camera_prim_path=camera_prim_path,
        resolution=resolution,
        save_path=save_path,
        per_frame_save_path=per_frame_save_path,
        num_frames=num_frames,
    )
    return await computer.compute()


def apply_coverage_colors(npz_path: str, mesh_prim_path: str):
    """Apply vertex colours from a saved coverage ``.npz`` to a mesh.

    Returns ``(n_red, n_blue)`` or ``None`` on failure.
    """
    _ensure_paths()
    from coverage_map_npz2meshColor import apply_vertex_colors_from_npz  # noqa: E402
    return apply_vertex_colors_from_npz(npz_path, mesh_prim_path)


def set_coverage_materials_visible(mesh_prim_path: str, visible: bool):
    """Toggle material bindings on a mesh to show coverage colours or materials."""
    _ensure_paths()
    from coverage_map_npz2meshColor import set_materials_visible  # noqa: E402
    set_materials_visible(mesh_prim_path, visible)
