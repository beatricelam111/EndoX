"""
Centreline Tool - Motion Path
================================
Create and manage BasisCurves motion paths in the Omniverse scene.

Can be used as:
  - Part of the centreline_tool package (import)
  - Standalone script in the Omniverse Script Editor (paste & run)

GUI button mapping:
  - "Show Motion Path"  -> MotionPath.create_from_npy(npy_path)
  - "Remove Motion Path" -> MotionPath.remove()
"""

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf

try:
    from .config import CentrelineConfig
except ImportError:
    from config import CentrelineConfig


class MotionPath:
    """Create and manage a BasisCurves motion path in the USD scene."""

    def __init__(self, config: CentrelineConfig = None):
        self.config = config or CentrelineConfig()

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_stage():
        return omni.usd.get_context().get_stage()

    @staticmethod
    def _add_endpoint_clamping(pts, curve_type, curve_basis):
        """
        Add phantom / repeated endpoints for cubic curves so the rendered
        curve passes exactly through the first and last control points.

          - bspline:    triple the first and last points
          - catmullRom: double the first and last points
          - bezier / linear: no change needed
        """
        if curve_type != "cubic" or len(pts) < 2:
            return pts

        first, last = pts[0], pts[-1]

        if curve_basis == "bspline":
            return [first, first] + list(pts) + [last, last]
        elif curve_basis == "catmullRom":
            return [first] + list(pts) + [last]

        return pts

    # ── public API ───────────────────────────────────────────────────────

    def create(self, points_array,
               curve_path=None, curve_type=None, curve_basis=None, width=None):
        """
        Create a BasisCurves prim from a numpy point array.

        Parameters
        ----------
        points_array : ndarray (N, 3) or list of (x, y, z)
        curve_path   : str   - USD prim path (default from config)
        curve_type   : str   - "cubic" or "linear"
        curve_basis  : str   - "bspline", "bezier", or "catmullRom"
        width        : float - display width

        Returns
        -------
        UsdGeom.BasisCurves - the created prim
        """
        stage = self._get_stage()

        path  = curve_path  or self.config.motion_path_full
        ctype = curve_type  or self.config.curve_type
        basis = curve_basis or self.config.curve_basis
        w     = width       if width is not None else self.config.curve_width

        # Ensure parent Xform exists
        root_path = self.config.motion_path_root
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            UsdGeom.Xform.Define(stage, root_path)

        # Build control points with clamping
        pts = list(points_array)
        pts = self._add_endpoint_clamping(pts, ctype, basis)

        usd_pts = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts]

        # Create BasisCurves
        curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(path))
        curves.CreatePointsAttr().Set(usd_pts)
        curves.CreateCurveVertexCountsAttr().Set([len(usd_pts)])
        curves.CreateTypeAttr().Set(ctype)
        curves.CreateBasisAttr().Set(basis)
        curves.CreateWidthsAttr().Set([w])

        return curves

    def create_from_npy(self, npy_path: str = None):
        """
        Load a .npy file and create the motion path in the scene.
        This is the main entry point for the "Show Motion Path" button.

        Parameters
        ----------
        npy_path : str or None - uses config default if None

        Returns
        -------
        UsdGeom.BasisCurves - the created prim
        """
        path = npy_path or self.config.centerline_path
        centerline = np.load(path)

        if len(centerline) < 2:
            raise ValueError("Centerline has fewer than 2 points.")

        return self.create(centerline)

    def remove(self):
        """Remove the motion path prim from the scene."""
        stage = self._get_stage()
        path = self.config.motion_path_full
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)
            return True
        return False

    def exists(self) -> bool:
        """Check if a motion path prim already exists."""
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(self.config.motion_path_full)
        return prim.IsValid()


# ============================================================================
# STANDALONE - paste into Omniverse Script Editor and run
# ============================================================================
if __name__ == "__main__":
    import carb

    # ── Configuration ────────────────────────────────────────────────────
    NPY_PATH         = r""
    MOTION_PATH_ROOT = "/World/Centerline"
    MOTION_PATH_NAME = "MotionPath"
    CURVE_TYPE       = "cubic"       # "cubic" or "linear"
    CURVE_BASIS      = "bspline"     # "bspline", "bezier", "catmullRom"
    CURVE_WIDTH      = 2.0
    # ─────────────────────────────────────────────────────────────────────

    cfg = CentrelineConfig()
    cfg.motion_path_root = MOTION_PATH_ROOT
    cfg.motion_path_name = MOTION_PATH_NAME
    cfg.curve_type       = CURVE_TYPE
    cfg.curve_basis      = CURVE_BASIS
    cfg.curve_width      = CURVE_WIDTH

    mp = MotionPath(cfg)
    curves = mp.create_from_npy(NPY_PATH)
    carb.log_info(f"Motion path created: {cfg.motion_path_full}")
