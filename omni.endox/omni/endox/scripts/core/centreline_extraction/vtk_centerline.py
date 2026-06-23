"""
VTK-only centreline extraction via Voronoi diagram of the Delaunay
tetrahedralization, faithfully replicating the VMTK algorithm:

    Antiga, L. (2003). "Computational Geometry for Patient-Specific
    Reconstruction and Meshing of Blood Vessels from MR and CT Angiography."
    IEEE Transactions on Medical Imaging, 21(2).

Ported from the VMTK C++ source (github.com/vmtk/vmtk):
  - vtkvmtkInternalTetrahedraExtractor  (normal dot-product inside/outside test)
  - vtkvmtkVoronoiDiagram3D            (circumcenters, poles, Voronoi topology)
  - vtkvmtkNonManifoldFastMarching      (Eikonal with 1/R cost, approx. via Dijkstra)
  - vtkvmtkSteepestDescentLineTracer    (backtracing via predecessor chain)

Requires only VTK and NumPy - both bundled with Omniverse Kit - so this
runs directly inside Kit without a conda / VMTK environment.

Usage::

    from vtk_centerline import VoronoiCenterline

    extractor = VoronoiCenterline(verbose=True)
    points = extractor.extract(polydata, source_pt, target_pt)
"""

import heapq
from collections import defaultdict

import numpy as np
import vtk
from vtk.util import numpy_support


# ═════════════════════════════════════════════════════════════════════════════
# SURFACE PREPARATION
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_normals(polydata):
    """Consistent outward-facing point normals via vtkPolyDataNormals.

    Matches VMTK: SplittingOff, AutoOrientNormalsOn, ConsistencyOn,
    ComputePointNormalsOn.
    """
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(polydata)
    nf.SplittingOff()
    nf.AutoOrientNormalsOn()
    nf.ConsistencyOn()
    nf.ComputePointNormalsOn()
    nf.Update()
    return nf.GetOutput()


def _subdivide_surface(polydata, min_points=1000, max_iters=3):
    """Loop-subdivide coarse meshes to ensure sufficient Delaunay density."""
    result = polydata
    for _ in range(max_iters):
        if result.GetNumberOfPoints() >= min_points:
            break
        sub = vtk.vtkLoopSubdivisionFilter()
        sub.SetInputData(result)
        sub.SetNumberOfSubdivisions(1)
        sub.Update()
        result = sub.GetOutput()
    return result


def _decimate_surface(polydata, target_n):
    """Reduce mesh to approximately *target_n* vertices."""
    n = polydata.GetNumberOfPoints()
    if n <= target_n:
        return polydata
    reduction = min(1.0 - target_n / n, 0.95)
    dec = vtk.vtkDecimatePro()
    dec.SetInputData(polydata)
    dec.SetTargetReduction(reduction)
    dec.PreserveTopologyOn()
    dec.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(dec.GetOutput())
    clean.Update()
    return clean.GetOutput()


# ═════════════════════════════════════════════════════════════════════════════
# CIRCUMSPHERE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def _circumspheres(tetras):
    """Vectorised circumsphere computation for every tetrahedron.

    Returns
    -------
    centers : ndarray (N, 3) - circumcentres
    radii   : ndarray (N,)   - circumradii
    ids     : ndarray (N, 4) - vertex point-IDs for each tet
    all_pts : ndarray (M, 3) - all Delaunay points
    """
    pts_vtk = tetras.GetPoints()
    all_pts = numpy_support.vtk_to_numpy(pts_vtk.GetData()).copy()

    n_cells = tetras.GetNumberOfCells()
    ids = np.empty((n_cells, 4), dtype=np.int64)
    for i in range(n_cells):
        cell = tetras.GetCell(i)
        for j in range(4):
            ids[i, j] = cell.GetPointId(j)

    p0 = all_pts[ids[:, 0]]
    p1 = all_pts[ids[:, 1]]
    p2 = all_pts[ids[:, 2]]
    p3 = all_pts[ids[:, 3]]

    a = p0 - p3
    b = p1 - p3
    c = p2 - p3

    M = 2.0 * np.stack([a, b, c], axis=1)                  # (N, 3, 3)
    rhs = np.stack([
        np.sum(a * (p0 + p3), axis=1),
        np.sum(b * (p1 + p3), axis=1),
        np.sum(c * (p2 + p3), axis=1),
    ], axis=1)                                              # (N, 3)

    dets = np.linalg.det(M)
    good = np.abs(dets) > 1e-14

    centers = np.empty_like(p0)
    radii = np.empty(n_cells)

    if np.any(good):
        sol = np.linalg.solve(M[good], rhs[good, :, np.newaxis])  # (K,3,1)
        centers[good] = sol[:, :, 0]
        radii[good] = np.linalg.norm(centers[good] - p0[good], axis=1)

    bad = ~good
    if np.any(bad):
        centers[bad] = (p0[bad] + p1[bad] + p2[bad] + p3[bad]) * 0.25
        radii[bad] = 0.0

    return centers, radii, ids, all_pts


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL TETRAHEDRA EXTRACTION  (port of vtkvmtkInternalTetrahedraExtractor)
# ═════════════════════════════════════════════════════════════════════════════

def _inside_mask(centers, surface):
    """Boolean mask - True for circumcentres inside the closed surface.

    Uses ``vtkSelectEnclosedPoints`` which handles concave surfaces
    correctly (unlike the dot-product test alone).
    """
    vtk_arr = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(centers, dtype=np.float64),
        deep=True,
        array_type=vtk.VTK_DOUBLE,
    )
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(vtk_arr)

    cloud = vtk.vtkPolyData()
    cloud.SetPoints(vtk_pts)

    sel = vtk.vtkSelectEnclosedPoints()
    sel.SetInputData(cloud)
    sel.SetSurfaceData(surface)
    sel.SetTolerance(0.0001)
    sel.Update()

    n = len(centers)
    output = sel.GetOutput()
    arr = output.GetPointData().GetArray("SelectedPoints")
    if arr is not None:
        mask = numpy_support.vtk_to_numpy(arr).astype(bool)
    else:
        mask = np.array([sel.IsInside(i) for i in range(n)], dtype=bool)

    sel.Complete()
    return mask


def _extract_internal_tetras(all_pts, ids, centers, normals_array):
    """Classify tetrahedra as internal using the VMTK normal dot-product test.

    For each tetrahedron the vector from circumcenter to each vertex is
    dotted with the outward surface normal at that vertex.  If all four
    dot products are positive the circumcenter lies on the interior side
    of every vertex, so the tetrahedron is internal.

    Parameters
    ----------
    all_pts       : (M, 3) all Delaunay points
    ids           : (N, 4) vertex point-IDs per tet
    centers       : (N, 3) circumcenters
    normals_array : (P, 3) outward normals for the first P surface points

    Returns
    -------
    keep : (N,) bool - True for internal tetrahedra
    """
    n_normals = len(normals_array)
    n_pts = len(all_pts)

    padded = np.zeros((n_pts, 3), dtype=np.float64)
    padded[:n_normals] = normals_array

    p0 = all_pts[ids[:, 0]]
    p1 = all_pts[ids[:, 1]]
    p2 = all_pts[ids[:, 2]]
    p3 = all_pts[ids[:, 3]]

    v0 = p0 - centers
    v1 = p1 - centers
    v2 = p2 - centers
    v3 = p3 - centers

    n0 = padded[ids[:, 0]]
    n1 = padded[ids[:, 1]]
    n2 = padded[ids[:, 2]]
    n3 = padded[ids[:, 3]]

    dot0 = np.sum(v0 * n0, axis=1)
    dot1 = np.sum(v1 * n1, axis=1)
    dot2 = np.sum(v2 * n2, axis=1)
    dot3 = np.sum(v3 * n3, axis=1)

    tol = 1e-12
    keep = (dot0 > tol) & (dot1 > tol) & (dot2 > tol) & (dot3 > tol)
    return keep


# ═════════════════════════════════════════════════════════════════════════════
# SMOOTHING
# ═════════════════════════════════════════════════════════════════════════════

def _refine_path(pts, surface, iterations=20, center_weight=0.3,
                 smooth_alpha=0.4):
    """Push path points toward the tube center via distance-field gradient
    ascent, interleaved with Laplacian smoothing.

    Each iteration:
      1. For every inner point, find the closest point on the surface
         (via vtkCellLocator - accurate to cell, not just vertex).
      2. Compute the wall-repulsion vector (projected onto the local
         cross-section plane) and push toward the centre.
      3. Apply one pass of Laplacian smoothing (endpoints pinned).
    """
    if len(pts) < 3:
        return pts.copy()

    cell_loc = vtk.vtkCellLocator()
    cell_loc.SetDataSet(surface)
    cell_loc.BuildLocator()

    out = pts.copy()
    n = len(out)
    closest_pt = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for _ in range(iterations):
        # Estimate local tube radius as the max wall distance
        wall_dists = np.empty(n)
        wall_dirs = np.zeros((n, 3))
        for i in range(n):
            cell_loc.FindClosestPoint(
                out[i].tolist(), closest_pt, cell_id, sub_id, dist2,
            )
            wall_pt = np.array(closest_pt)
            wall_vec = out[i] - wall_pt
            wall_dists[i] = np.linalg.norm(wall_vec)
            if wall_dists[i] > 1e-12:
                wall_dirs[i] = wall_vec / wall_dists[i]

        r_est = np.max(wall_dists[1:-1]) if n > 2 else 1.0

        # ── centering: push off-centre points toward the middle ──
        for i in range(1, n - 1):
            if wall_dists[i] < 1e-12:
                continue

            deficit = max(r_est - wall_dists[i], 0.0)
            if deficit < 1e-12:
                continue

            d = wall_dirs[i].copy()

            tangent = out[i + 1] - out[i - 1]
            t_len = np.linalg.norm(tangent)
            if t_len > 1e-12:
                tangent /= t_len
                d -= np.dot(d, tangent) * tangent
                d_len = np.linalg.norm(d)
                if d_len < 1e-12:
                    continue
                d /= d_len

            out[i] += d * center_weight * deficit

        # ── Taubin smoothing (non-shrinking, endpoints pinned) ──
        lam = smooth_alpha
        mu = -smooth_alpha - 0.01
        for i in range(1, n - 1):
            laplacian = (out[i - 1] + out[i + 1]) * 0.5 - out[i]
            out[i] += lam * laplacian
        for i in range(1, n - 1):
            laplacian = (out[i - 1] + out[i + 1]) * 0.5 - out[i]
            out[i] += mu * laplacian

    return out


def _smooth_polyline(pts, iterations=20, alpha=0.5):
    """Laplacian smoothing of a 3-D polyline (endpoints pinned)."""
    if len(pts) < 3:
        return pts.copy()
    out = pts.copy()
    for _ in range(iterations):
        new = out.copy()
        for i in range(1, len(out) - 1):
            new[i] = out[i] * (1.0 - alpha) + (out[i - 1] + out[i + 1]) * 0.5 * alpha
        out = new
    return out


# ═════════════════════════════════════════════════════════════════════════════
# VORONOI CENTERLINE CLASS
# ═════════════════════════════════════════════════════════════════════════════

class VoronoiCenterline:
    """Centreline extraction via Voronoi diagram of the Delaunay
    tetrahedralization (Antiga, 2003).  VTK + NumPy only.

    Faithfully replicates the VMTK C++ pipeline:

    1. Compute outward surface normals (vtkPolyDataNormals)
    2. Delaunay 3-D tetrahedralization
    3. Extract internal tetrahedra via normal dot-product test
       (port of vtkvmtkInternalTetrahedraExtractor)
    4. Build Voronoi diagram: circumcenters as vertices, face-adjacency
       as edges, circumradii as scalars
       (port of vtkvmtkVoronoiDiagram3D)
    5. Compute poles: for each surface point, the internal tet with
       the largest circumradius
    6. Dijkstra with cost = 1/R on the Voronoi graph
       (approximation of vtkvmtkNonManifoldFastMarching)
    7. Laplacian-smooth the extracted polyline

    Parameters
    ----------
    max_surface_points : int
        Decimate input mesh to at most this many vertices.
    smooth_iterations : int
        Laplacian smoothing passes on the output polyline.
    smooth_alpha : float (0-1)
        Blend factor for smoothing.
    verbose : bool
        Print step-by-step progress.
    """

    def __init__(self, max_surface_points=12000, smooth_iterations=30,
                 smooth_alpha=0.5, verbose=False):
        self.max_surface_points = max_surface_points
        self.smooth_iterations = smooth_iterations
        self.smooth_alpha = smooth_alpha
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ── poles (port of vtkvmtkVoronoiDiagram3D pole computation) ──────

    @staticmethod
    def _compute_poles(tetras, keep_mask, radii, n_surface_pts):
        """For each surface point find its Voronoi pole.

        The pole is the circumcenter of the internal tetrahedron with
        the largest circumradius among all internal tets incident to
        that surface point.  This matches VMTK's VoronoiDiagram3D.

        Returns
        -------
        pole_cell : dict  surface-point-id -> cell-index (into all tets)
        """
        tetras.BuildLinks()
        pole_cell = {}
        id_list = vtk.vtkIdList()

        for pt_id in range(n_surface_pts):
            id_list.Initialize()
            tetras.GetPointCells(pt_id, id_list)

            best_cell = -1
            best_r = 0.0
            for j in range(id_list.GetNumberOfIds()):
                ci = id_list.GetId(j)
                if not keep_mask[ci]:
                    continue
                r = radii[ci]
                if r - best_r > 1e-12:
                    best_r = r
                    best_cell = ci
            if best_cell >= 0:
                pole_cell[pt_id] = best_cell
        return pole_cell

    # ── face adjacency ────────────────────────────────────────────────

    @staticmethod
    def _face_adjacency(tetras, keep_mask, cell_to_vi):
        """Build adjacency between Voronoi vertices whose Delaunay
        tetrahedra share a triangular face.

        Parameters
        ----------
        tetras      : vtkUnstructuredGrid
        keep_mask   : (N,) bool - True for internal tets
        cell_to_vi  : dict  cell-index -> Voronoi-vertex-index

        Returns
        -------
        adj : defaultdict(set)  vi -> set of neighbour vi
        """
        face_map = defaultdict(list)
        n = tetras.GetNumberOfCells()

        for ci in range(n):
            if not keep_mask[ci]:
                continue
            cell = tetras.GetCell(ci)
            vid = [cell.GetPointId(j) for j in range(4)]
            for tri in (
                (vid[0], vid[1], vid[2]),
                (vid[0], vid[1], vid[3]),
                (vid[0], vid[2], vid[3]),
                (vid[1], vid[2], vid[3]),
            ):
                face_map[tuple(sorted(tri))].append(ci)

        adj = defaultdict(set)
        for cells in face_map.values():
            if len(cells) == 2:
                a = cell_to_vi.get(cells[0])
                b = cell_to_vi.get(cells[1])
                if a is not None and b is not None and a != b:
                    adj[a].add(b)
                    adj[b].add(a)
        return adj

    # ── KNN fallback ─────────────────────────────────────────────────

    @staticmethod
    def _knn_edges(adj, poles, k=8):
        """Add k-nearest-neighbour edges as fallback for sparse graphs."""
        n = len(poles)
        k = min(k, n - 1)
        if k < 1:
            return
        for i in range(n):
            dists = np.linalg.norm(poles - poles[i], axis=1)
            nearest = np.argsort(dists)[1:k + 1]
            for j in nearest:
                adj[i].add(int(j))
                adj[int(j)].add(i)

    # ── seed selection (port of FindVoronoiSeeds) ────────────────────

    @staticmethod
    def _find_seed(surface, user_point, pole_cell, cell_to_vi, voronoi_pts):
        """Map a user-specified 3D point to a Voronoi vertex index.

        Strategy (matching VMTK):
        1. Find the nearest surface vertex to user_point.
        2. Look up that vertex's pole (max-R tet).
        3. Map the pole's cell-index to a Voronoi vertex index.

        Falls back to nearest Voronoi vertex if pole lookup fails.
        """
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(surface)
        locator.BuildLocator()
        nearest_pt = locator.FindClosestPoint(user_point)

        if nearest_pt in pole_cell:
            ci = pole_cell[nearest_pt]
            if ci in cell_to_vi:
                return cell_to_vi[ci]

        dists = np.linalg.norm(voronoi_pts - np.asarray(user_point), axis=1)
        return int(np.argmin(dists))

    # ── Dijkstra with 1/R cost ───────────────────────────────────────

    @staticmethod
    def _dijkstra(adj, poles, radii, start, end):
        """Shortest path on the Voronoi graph with cost = 1/R^4.

        Edge weight = edge_length / mean_R^4.  The fourth-power reciprocal
        radius very strongly penalises near-wall paths, compensating for
        the fact that we traverse a 3-D volume graph rather than the 2-D
        Voronoi surface that VMTK's Fast Marching operates on.
        """
        EPS = 1e-30
        n = len(poles)
        dist = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=np.intp)
        dist[start] = 0.0

        heap = [(0.0, start)]
        visited = np.zeros(n, dtype=bool)

        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            if u == end:
                break
            for v in adj.get(u, set()):
                if visited[v]:
                    continue
                edge_len = float(np.linalg.norm(poles[u] - poles[v]))
                mean_r = 0.5 * (radii[u] + radii[v])
                r4 = max(mean_r ** 4, EPS)
                w = edge_len / r4
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if not visited[end]:
            return None

        path = []
        i = end
        while i != -1:
            path.append(i)
            i = int(prev[i])
        path.reverse()
        return poles[path]

    # ── main entry point ─────────────────────────────────────────────

    def extract(self, polydata, source_point, target_point):
        """Extract centreline from a closed (watertight) surface mesh.

        Parameters
        ----------
        polydata     : vtkPolyData - watertight triangle mesh
        source_point : array-like (3,) - inlet position
        target_point : array-like (3,) - outlet position

        Returns
        -------
        ndarray (N, 3) - centreline points, or None on failure
        """
        src = np.asarray(source_point, dtype=np.float64)
        tgt = np.asarray(target_point, dtype=np.float64)

        # ── 1. Prepare surface ─────────────────────────────────────
        self._log("[Voronoi] 1/7  Preparing surface ...")
        n_orig = polydata.GetNumberOfPoints()
        work = _subdivide_surface(polydata, min_points=1000)
        work = _decimate_surface(work, self.max_surface_points)
        surface = _ensure_normals(work)
        n_surface = surface.GetNumberOfPoints()
        self._log(f"               {n_orig} -> {n_surface} vertices")

        normals_vtk = surface.GetPointData().GetNormals()
        if normals_vtk is None:
            self._log("[Voronoi] FAIL  could not compute surface normals")
            return None
        normals_array = numpy_support.vtk_to_numpy(normals_vtk).copy()

        # ── 2. Delaunay 3D ─────────────────────────────────────────
        self._log("[Voronoi] 2/7  Delaunay 3D tetrahedralization ...")
        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(surface)
        delaunay.SetTolerance(1e-3)
        delaunay.Update()
        tetras = delaunay.GetOutput()

        n_cells = tetras.GetNumberOfCells()
        self._log(f"               {n_cells} tetrahedra")
        if n_cells == 0:
            self._log("[Voronoi] FAIL  Delaunay produced 0 tetrahedra")
            return None

        # ── 3. Circumspheres ──────────────────────────────────────
        self._log("[Voronoi] 3/7  Computing circumspheres ...")
        centers, radii, tet_ids, all_pts = _circumspheres(tetras)

        # ── 4. Internal tetrahedra extraction ─────────────────────
        self._log("[Voronoi] 4/7  Extracting internal tetrahedra ...")

        # VMTK dot-product test (fast, handles convex shapes well)
        dot_mask = _extract_internal_tetras(
            all_pts, tet_ids, centers, normals_array,
        )
        n_dot = int(np.sum(dot_mask))
        self._log(f"               {n_dot} / {n_cells} pass dot-product test")

        # Geometric enclosure test (robust for concave / curved shapes)
        enc_mask = _inside_mask(centers, surface)
        n_enc = int(np.sum(enc_mask))
        self._log(f"               {n_enc} / {n_cells} inside surface (enclosure)")

        keep_mask = dot_mask & enc_mask
        int_indices = np.where(keep_mask)[0]
        n_int = len(int_indices)
        self._log(f"               {n_int} / {n_cells} internal (intersection)")

        if n_int < 2:
            self._log("[Voronoi] FAIL  fewer than 2 internal tetrahedra")
            return None

        cell_to_vi = {int(ci): vi for vi, ci in enumerate(int_indices)}
        voronoi_pts = centers[int_indices]

        # Compute true wall-distance for each circumcenter rather
        # than using the circumradius (which can be larger than the
        # actual inscribed sphere radius for large spanning tets).
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(surface)
        locator.BuildLocator()
        surf_pts_np = numpy_support.vtk_to_numpy(
            surface.GetPoints().GetData(),
        )
        wall_dist = np.empty(n_int, dtype=np.float64)
        for vi in range(n_int):
            nearest = locator.FindClosestPoint(voronoi_pts[vi])
            wall_dist[vi] = np.linalg.norm(
                voronoi_pts[vi] - surf_pts_np[nearest],
            )
        voronoi_r = wall_dist
        self._log(f"               wall-distance range: "
                  f"{voronoi_r.min():.2f} - {voronoi_r.max():.2f}")

        # ── 5. Poles + seed selection ─────────────────────────────
        self._log("[Voronoi] 5/7  Computing poles & seeds ...")
        pole_cell = self._compute_poles(
            tetras, keep_mask, radii, n_surface,
        )
        self._log(f"               {len(pole_cell)} surface points have poles")

        si = self._find_seed(surface, src, pole_cell, cell_to_vi, voronoi_pts)
        ti = self._find_seed(surface, tgt, pole_cell, cell_to_vi, voronoi_pts)
        self._log(f"               source seed = {si}  target seed = {ti}")

        if si == ti:
            self._log("[Voronoi] WARN  source and target map to same Voronoi vertex")
            return voronoi_pts[si].reshape(1, 3)

        # ── 6. Adjacency + Dijkstra with 1/R ─────────────────────
        self._log("[Voronoi] 6/7  Building graph & Dijkstra (cost = 1/R) ...")
        adj = self._face_adjacency(tetras, keep_mask, cell_to_vi)

        avg_deg = (np.mean([len(adj.get(i, set())) for i in range(n_int)])
                   if n_int else 0.0)
        self._log(f"               avg degree = {avg_deg:.1f}")

        if avg_deg < 2.0:
            self._log("               Sparse graph - augmenting with KNN ...")
            self._knn_edges(adj, voronoi_pts, k=8)

        path = self._dijkstra(adj, voronoi_pts, voronoi_r, si, ti)

        if path is None:
            self._log("               No path - retrying with denser KNN ...")
            self._knn_edges(adj, voronoi_pts, k=16)
            path = self._dijkstra(adj, voronoi_pts, voronoi_r, si, ti)

        if path is None:
            self._log("[Voronoi] FAIL  no path between source and target")
            return None

        self._log(f"               raw path: {len(path)} points")

        # ── 7. Refine: centre + smooth ───────────────────────────
        if self.smooth_iterations > 0 and len(path) > 2:
            self._log(f"[Voronoi] 7/7  Centering + smoothing "
                      f"({self.smooth_iterations} iters) ...")
            path = _refine_path(
                path, surface,
                iterations=self.smooth_iterations,
                smooth_alpha=self.smooth_alpha,
            )
        else:
            self._log("[Voronoi] 7/7  Smoothing skipped")

        self._log(f"[Voronoi] Done - {len(path)} centreline points")
        return path


# ═════════════════════════════════════════════════════════════════════════════
# STANDALONE CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="VTK-only centreline extraction (Voronoi / Antiga 2003)")
    parser.add_argument("--input", required=True,
                        help=".npz scene data (vertices, faces, inlet_pos, outlet_pos)")
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--max-points", type=int, default=12000,
                        help="Decimate surface to this many vertices (default: 12000)")
    parser.add_argument("--smooth", type=int, default=30,
                        help="Laplacian smoothing iterations (default: 30)")
    args = parser.parse_args()

    data = np.load(args.input)
    verts = data["vertices"]
    faces = data["faces"]
    inlet = data["inlet_pos"]
    outlet = data["outlet_pos"]

    vtk_points = vtk.vtkPoints()
    for v in verts:
        vtk_points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
    vtk_polys = vtk.vtkCellArray()
    for f in faces:
        vtk_polys.InsertNextCell(3)
        for idx in f:
            vtk_polys.InsertCellPoint(int(idx))
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_points)
    pd.SetPolys(vtk_polys)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(pd)
    cleaner.Update()

    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(cleaner.GetOutput())
    filler.SetHoleSize(1e6)
    filler.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(filler.GetOutput())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    watertight = normals.GetOutput()

    extractor = VoronoiCenterline(
        max_surface_points=args.max_points,
        smooth_iterations=args.smooth,
        verbose=True,
    )
    result = extractor.extract(watertight, inlet, outlet)

    if result is None:
        print("[ERROR] Centreline extraction failed.", flush=True)
        raise SystemExit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, result)
    print(f"Saved {len(result)} points -> {args.output}")
