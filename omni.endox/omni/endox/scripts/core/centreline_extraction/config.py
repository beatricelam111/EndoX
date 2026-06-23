"""
Centreline Tool - Configuration
"""

import os


def _default_output_dir():
    """Resolve <package>/results at call time (survives importlib.reload)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class CentrelineConfig:
    """
    Central configuration for the centreline extraction pipeline.

    Each attribute maps to a GUI control (text field, checkbox, slider, etc.).
    Modify defaults here or override at runtime.
    """

    def __init__(self):
        # ── Scene Paths ──────────────────────────────────────────────────
        self.mesh_prim_path: str = ""           # USD path to the organ mesh prim
        self.inlet_prim_path: str = ""          # USD path to inlet sphere (empty = auto-detect)
        self.outlet_prim_path: str = ""         # USD path to outlet sphere (empty = auto-detect)

        # ── Sphere Markers ───────────────────────────────────────────────
        self.sphere_radius: float = 5.0         # Default sphere marker radius
        self.inlet_color: tuple = (0.0, 1.0, 0.0)    # Green
        self.outlet_color: tuple = (1.0, 0.0, 0.0)   # Red

        # ── Output ───────────────────────────────────────────────────────
        self.output_dir: str = _default_output_dir()
        self.scene_data_filename: str = "scene_data.npz"
        self.centerline_filename: str = "centerline.npy"
        self.centerline_raw_filename: str = "centerline_raw.npy"

        # ── VMTK / Centerline ───────────────────────────────────────────
        self.conda_env: str = "vmtk"             # Name of conda env with VMTK
        self.vmtk_python_path: str = ""         # Override: full path to conda python
        self.make_watertight: bool = True        # Fill holes before VMTK
        self.fill_hole_size: float = 1e6         # Max hole size to fill

        # ── Simplification ───────────────────────────────────────────────
        self.simplify: bool = True               # Reduce point count
        self.percent_to_keep: float = 0.10       # Fraction of points to keep (0.0-1.0)

        # ── Motion Path (BasisCurves) ────────────────────────────────────
        self.motion_path_root: str = "/World/Centerline"
        self.motion_path_name: str = "MotionPath"
        self.curve_type: str = "cubic"           # "cubic" or "linear"
        self.curve_basis: str = "bspline"        # "bspline", "bezier", "catmullRom"
        self.curve_width: float = 2.0

    # ── Derived paths ────────────────────────────────────────────────────

    def _resolve_output_dir(self):
        """Return output_dir, falling back to <package>/results if empty."""
        return self.output_dir or _default_output_dir()

    @property
    def scene_data_path(self) -> str:
        return os.path.join(self._resolve_output_dir(), self.scene_data_filename)

    @property
    def centerline_path(self) -> str:
        return os.path.join(self._resolve_output_dir(), self.centerline_filename)

    @property
    def centerline_raw_path(self) -> str:
        return os.path.join(self._resolve_output_dir(), self.centerline_raw_filename)

    @property
    def motion_path_full(self) -> str:
        return f"{self.motion_path_root}/{self.motion_path_name}"

    def __repr__(self):
        lines = ["CentrelineConfig("]
        for k, v in self.__dict__.items():
            lines.append(f"  {k} = {v!r},")
        lines.append(")")
        return "\n".join(lines)
