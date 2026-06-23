"""
Centreline Tool - Centerline Computer
========================================
Compute the volumetric centerline with post-processing
(orientation, clamping, simplification).

Extraction back-ends (tried in order):
  1. VMTK vmtkCenterlines (if installed - typically via conda)
  2. VTK-only Voronoi diagram (Antiga, 2003) - no external dependency

This module can run EITHER inside Omniverse Kit (where the VTK-only
back-end is used automatically) OR in an external Python environment.

Can be used as:
  - Part of the centreline_tool package (import)
  - Standalone script (python centerline_computer.py)

GUI button mapping:
  - "Compute Centerline" -> CenterlineComputer.compute()
"""

import os
import sys
import numpy as np

try:
    from .config import CentrelineConfig
except ImportError:
    from config import CentrelineConfig


# ═════════════════════════════════════════════════════════════════════════════
# MESH CONVERSION
# ═════════════════════════════════════════════════════════════════════════════

class MeshConverter:
    """Convert numpy mesh arrays to a cleaned, watertight VTK PolyData."""

    @staticmethod
    def to_vtk_polydata(vertices, faces, make_watertight=True):
        """
        Parameters
        ----------
        vertices       : ndarray (N, 3)
        faces          : ndarray (M, 3) - triangulated
        make_watertight: bool - fill holes to close inlet/outlet openings

        Returns
        -------
        vtkPolyData - cleaned and optionally watertight
        """
        import vtk

        vtk_points = vtk.vtkPoints()
        for v in vertices:
            vtk_points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))

        vtk_polys = vtk.vtkCellArray()
        for f in faces:
            vtk_polys.InsertNextCell(3)
            vtk_polys.InsertCellPoint(int(f[0]))
            vtk_polys.InsertCellPoint(int(f[1]))
            vtk_polys.InsertCellPoint(int(f[2]))

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_polys)

        # Clean
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        cleaner.Update()
        result = cleaner.GetOutput()

        if not make_watertight:
            return result

        # Fill holes
        filler = vtk.vtkFillHolesFilter()
        filler.SetInputData(result)
        filler.SetHoleSize(1e6)
        filler.Update()
        filled = filler.GetOutput()

        # Consistent normals
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(filled)
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()

        return normals.GetOutput()


# ═════════════════════════════════════════════════════════════════════════════
# VMTK PATCH
# ═════════════════════════════════════════════════════════════════════════════

class VMTKPatcher:
    """Fix Python 2 -> 3 integer division bugs in VMTK 1.4.0."""

    @staticmethod
    def patch():
        """
        VMTK 1.4.0 uses ``range(len(…)/3)`` which fails in Python 3
        because ``/`` returns float.  This patches the installed file
        to use ``//`` instead.  Safe to call multiple times.
        """
        try:
            import vmtk.vmtkcenterlines as mod
            fpath = mod.__file__
            if fpath.endswith('.pyc'):
                fpath = fpath[:-1]

            with open(fpath, 'r') as f:
                src = f.read()

            replacements = [
                ('range(len(self.SourcePoints)/3)',   'range(len(self.SourcePoints)//3)'),
                ('range(len(self.TargetPoints)/3)',   'range(len(self.TargetPoints)//3)'),
                ('range(len(self.SourcePoints) / 3)', 'range(len(self.SourcePoints) // 3)'),
                ('range(len(self.TargetPoints) / 3)', 'range(len(self.TargetPoints) // 3)'),
            ]

            patched = src
            count = 0
            for old, new in replacements:
                if old in patched:
                    patched = patched.replace(old, new)
                    count += 1

            if count > 0:
                with open(fpath, 'w') as f:
                    f.write(patched)
                import importlib
                importlib.reload(mod)

        except Exception:
            pass  # non-critical - VMTK may already be patched or absent


# ═════════════════════════════════════════════════════════════════════════════
# INTERIOR CLAMPING
# ═════════════════════════════════════════════════════════════════════════════

class InteriorClamper:
    """Fix centerline points that escape outside the mesh boundary."""

    def __init__(self, polydata):
        import vtk
        self._obb = vtk.vtkOBBTree()
        self._obb.SetDataSet(polydata)
        self._obb.SetTolerance(1e-6)
        self._obb.BuildLocator()

    def is_inside(self, point):
        """
        Multi-ray majority-vote inside/outside test.
        Robust for non-watertight meshes (13 rays, majority vote).
        """
        import vtk

        directions = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1), (1, 1, 1), (-1, -1, -1),
            (1, -1, 1), (-1, 1, -1), (1, 1, -1), (-1, -1, 1),
            (0.577, 0.817, 0.317),
        ]
        p = point.tolist()
        votes = 0
        for d in directions:
            end = [p[0] + d[0] * 1e6, p[1] + d[1] * 1e6, p[2] + d[2] * 1e6]
            pts = vtk.vtkPoints()
            ids = vtk.vtkIdList()
            self._obb.IntersectWithLine(p, end, pts, ids)
            if pts.GetNumberOfPoints() % 2 == 1:
                votes += 1

        return votes > len(directions) / 2

    def clamp(self, centerline, max_passes=3):
        """
        Fix outlier points by replacing them with neighbour midpoints.
        First and last points (inlet/outlet) are never modified.

        Returns
        -------
        ndarray (N, 3) - clamped centerline
        """
        pts = np.copy(centerline)
        total_fixed = 0

        for _pass in range(max_passes):
            fixed = 0
            for i in range(1, len(pts) - 1):
                if not self.is_inside(pts[i]):
                    pts[i] = (pts[i - 1] + pts[i + 1]) / 2.0
                    fixed += 1
            total_fixed += fixed
            if fixed == 0:
                break

        return pts


# ═════════════════════════════════════════════════════════════════════════════
# SIMPLIFICATION (Douglas-Peucker)
# ═════════════════════════════════════════════════════════════════════════════

class CurveSimplifier:
    """Reduce curve point count using adaptive Douglas-Peucker."""

    @staticmethod
    def _point_to_line_distance(point, start, end):
        line = end - start
        vec = point - start
        length_sq = np.dot(line, line)
        if length_sq == 0:
            return np.linalg.norm(vec)
        t = max(0, min(1, np.dot(vec, line) / length_sq))
        return np.linalg.norm(point - (start + t * line))

    @classmethod
    def _douglas_peucker(cls, points, epsilon):
        if len(points) <= 2:
            return points
        start, end = points[0], points[-1]
        max_d, max_i = 0, 0
        for i in range(1, len(points) - 1):
            d = cls._point_to_line_distance(points[i], start, end)
            if d > max_d:
                max_d, max_i = d, i
        if max_d <= epsilon:
            return np.array([start, end])
        left = cls._douglas_peucker(points[:max_i + 1], epsilon)
        right = cls._douglas_peucker(points[max_i:], epsilon)
        return np.vstack([left[:-1], right])

    @classmethod
    def simplify(cls, points, percent_to_keep=0.10):
        """
        Reduce point count to ~percent_to_keep of original.

        Parameters
        ----------
        points          : ndarray (N, 3)
        percent_to_keep : float (0.0-1.0)

        Returns
        -------
        ndarray (M, 3) - simplified curve (M <= N)
        """
        target = max(2, int(len(points) * percent_to_keep))
        if target >= len(points):
            return points.copy()

        bbox = np.ptp(points, axis=0)
        eps_max = np.linalg.norm(bbox)
        eps_min = 0.0
        best = points

        for _ in range(20):
            eps = (eps_min + eps_max) / 2
            result = cls._douglas_peucker(points, eps)
            if len(result) == target:
                return result
            elif len(result) > target:
                eps_min = eps
            else:
                eps_max = eps
                best = result

        return best


# ═════════════════════════════════════════════════════════════════════════════
# CENTERLINE COMPUTER (main class)
# ═════════════════════════════════════════════════════════════════════════════

class CenterlineComputer:
    """
    Compute the centreline of a tubular mesh.

    Extraction back-ends (automatic selection):
      - VMTK ``vmtkCenterlines`` - used when VMTK is importable
      - VTK-only Voronoi diagram (Antiga, 2003) - automatic fallback

    Full pipeline:
      1. Load scene data (.npz)
      2. Build watertight VTK PolyData
      3. Centreline extraction (VMTK -> VTK-only fallback)
      4. Orient (inlet -> outlet)
      5. Extend endpoints to exact marker positions
      6. Clamp outliers to mesh interior
      7. Simplify (Douglas-Peucker)
      8. Save .npy

    GUI button mapping:
      - "Compute Centerline" -> CenterlineComputer.compute()
    """

    def __init__(self, config: CentrelineConfig = None):
        self.config = config or CentrelineConfig()

        # Populated after compute()
        self.raw_centerline: np.ndarray = None
        self.simplified_centerline: np.ndarray = None

    # ── VMTK extraction ─────────────────────────────────────────────────

    @staticmethod
    def _vmtk_centerline(polydata, source_point, target_point):
        """Run VMTK vmtkCenterlines on a vtkPolyData surface."""
        from vmtk import vmtkscripts

        cl = vmtkscripts.vmtkCenterlines()
        cl.Surface = polydata
        cl.SeedSelectorName = "pointlist"
        cl.SourcePoints = [float(source_point[0]), float(source_point[1]),
                           float(source_point[2])]
        cl.TargetPoints = [float(target_point[0]), float(target_point[1]),
                           float(target_point[2])]
        cl.AppendEndPoints = 1
        cl.Execute()

        cl_pd = cl.Centerlines
        n = cl_pd.GetNumberOfPoints()
        if n == 0:
            return None

        pts = np.zeros((n, 3))
        for i in range(n):
            pts[i] = cl_pd.GetPoint(i)
        return pts

    # ── main entry point ────────────────────────────────────────────────

    def compute(self, input_npz: str = None, output_npy: str = None,
                verbose: bool = False):
        """
        Run the full centreline pipeline.

        Parameters
        ----------
        input_npz  : str or None - path to .npz from SceneExtractor
        output_npy : str or None - path for output .npy
        verbose    : bool - print progress to stdout

        Returns
        -------
        ndarray (M, 3) - the simplified centreline
        """
        def log(msg):
            if verbose:
                print(msg, flush=True)

        npz_path = input_npz or self.config.scene_data_path
        npy_path = output_npy or self.config.centerline_path

        # 1. Load scene data
        log(f"[1/8] Loading scene data: {npz_path}")
        data = np.load(npz_path)
        vertices = data["vertices"]
        faces = data["faces"]
        inlet_pos = data["inlet_pos"]
        outlet_pos = data["outlet_pos"]
        log(f"      Mesh: {len(vertices)} vertices, {len(faces)} faces")

        # 2. Build watertight VTK PolyData
        log("[2/8] Building watertight VTK mesh ...")
        polydata = MeshConverter.to_vtk_polydata(
            vertices, faces,
            make_watertight=self.config.make_watertight,
        )

        # 3. Centreline extraction (VMTK -> VTK-only Voronoi fallback)
        _vmtk_ok = True
        try:
            from vmtk import vmtkscripts  # noqa: F401
        except ImportError:
            _vmtk_ok = False

        if _vmtk_ok:
            log("[3/8] Running VMTK centreline extraction ...")
            VMTKPatcher.patch()
            centerline = self._vmtk_centerline(polydata, inlet_pos, outlet_pos)
        else:
            log("[3/8] VMTK not available - using VTK-only Voronoi extraction (Antiga 2003) ...")
            try:
                from .vtk_centerline import VoronoiCenterline
            except ImportError:
                from vtk_centerline import VoronoiCenterline
            extractor = VoronoiCenterline(verbose=verbose)
            centerline = extractor.extract(polydata, inlet_pos, outlet_pos)

        if centerline is None or len(centerline) < 2:
            raise RuntimeError("Centreline extraction returned empty result")
        log(f"      Extraction returned {len(centerline)} points")

        # 4. Orient: ensure inlet -> outlet
        log("[4/8] Orienting centreline (inlet -> outlet) ...")
        if np.linalg.norm(centerline[-1] - inlet_pos) < \
           np.linalg.norm(centerline[0] - inlet_pos):
            centerline = centerline[::-1].copy()

        # 5. Extend endpoints to exact marker positions
        log("[5/8] Extending endpoints to marker positions ...")
        centerline = np.vstack([
            inlet_pos.reshape(1, 3),
            centerline,
            outlet_pos.reshape(1, 3),
        ])

        self.raw_centerline = centerline.copy()

        # 6. Clamp outliers
        log("[6/8] Clamping outlier points to mesh interior ...")
        clamper = InteriorClamper(polydata)
        centerline = clamper.clamp(centerline)

        # 7. Simplify
        if self.config.simplify and self.config.percent_to_keep < 1.0:
            log(f"[7/8] Simplifying (keeping ~{self.config.percent_to_keep:.0%}) ...")
            simplified = CurveSimplifier.simplify(
                centerline, self.config.percent_to_keep
            )
        else:
            log("[7/8] Simplification skipped")
            simplified = centerline.copy()

        self.simplified_centerline = simplified

        # 8. Save
        log(f"[8/8] Saving: {npy_path}")
        out_dir = os.path.dirname(npy_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        np.save(npy_path, simplified)
        raw_path = os.path.join(out_dir, self.config.centerline_raw_filename)
        np.save(raw_path, self.raw_centerline)

        log(f"      Done - {len(self.raw_centerline)} raw pts -> {len(simplified)} simplified pts")
        return simplified


# ============================================================================
# STANDALONE - works with or without VMTK:
#   python centerline_computer.py               (VTK-only Voronoi fallback)
#   conda activate vmtk && python centerline_computer.py   (uses VMTK if available)
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute centreline (VMTK if available, else VTK-only Voronoi)")
    parser.add_argument("--input",  type=str, default=None, help=".npz input path")
    parser.add_argument("--output", type=str, default=None, help=".npy output path")
    args = parser.parse_args()

    cfg = CentrelineConfig()
    computer = CenterlineComputer(cfg)

    try:
        computer.compute(
            input_npz=args.input,
            output_npy=args.output,
            verbose=True,
        )
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
