"""
Centreline Tool - Scene Extractor
====================================
Extract mesh geometry and marker positions from the Omniverse scene
and pack them into a single .npz file for external processing.

Can be used as:
  - Part of the centreline_tool package (import)
  - Standalone script in the Omniverse Script Editor (paste & run)

GUI button mapping:
  - "Extract Scene Data" -> SceneExtractor.extract_and_save()
"""

import os
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf

try:
    from .config import CentrelineConfig
    from .sphere_marker import SphereMarker
except ImportError:
    from config import CentrelineConfig
    from sphere_marker import SphereMarker


class SceneExtractor:
    """Extracts mesh geometry and inlet/outlet positions from the USD stage."""

    def __init__(self, config: CentrelineConfig = None):
        self.config = config or CentrelineConfig()
        self._sphere_marker = SphereMarker(self.config)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_stage():
        return omni.usd.get_context().get_stage()

    @staticmethod
    def _find_mesh_prim(prim_path: str):
        """Locate a Mesh prim at or under *prim_path*."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        if prim.GetTypeName() == "Mesh":
            return prim
        for child in Usd.PrimRange(prim):
            if child.GetTypeName() == "Mesh":
                return child
        return None

    # ── public API ───────────────────────────────────────────────────────

    def extract_mesh(self, mesh_prim_path: str = None):
        """
        Extract vertices and triangulated faces from a USD mesh prim.
        Applies the full world transform so coordinates are in world space.

        Parameters
        ----------
        mesh_prim_path : str or None - uses config default if None

        Returns
        -------
        (vertices, faces) : (ndarray (N,3), ndarray (M,3)) or (None, None)
        """
        path = mesh_prim_path or self.config.mesh_prim_path
        mesh_prim = self._find_mesh_prim(path)
        if mesh_prim is None:
            raise ValueError(f"No Mesh prim found at or under: {path}")

        mesh = UsdGeom.Mesh(mesh_prim)
        usd_points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()

        if not usd_points or not face_counts or not face_indices:
            raise ValueError(f"Mesh has no geometry data: {mesh_prim.GetPath()}")

        # World transform (vectorised)
        xformable = UsdGeom.Xformable(mesh_prim)
        world_mat = np.array(
            xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )

        pts = np.array(usd_points, dtype=np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        world_pts = (np.hstack([pts, ones]) @ world_mat)[:, :3]

        # Triangulate
        triangles = []
        idx = 0
        for count in face_counts:
            c = int(count)
            verts = [int(face_indices[idx + j]) for j in range(c)]
            if c == 3:
                triangles.append(verts)
            elif c == 4:
                triangles.append([verts[0], verts[1], verts[2]])
                triangles.append([verts[0], verts[2], verts[3]])
            else:
                for j in range(1, c - 1):
                    triangles.append([verts[0], verts[j], verts[j + 1]])
            idx += c

        faces_np = np.array(triangles, dtype=np.int64)
        return world_pts, faces_np

    def extract_positions(self):
        """
        Get inlet and outlet world positions.

        Returns
        -------
        (inlet_pos, outlet_pos) : (ndarray (3,), ndarray (3,))

        Raises
        ------
        ValueError if inlet or outlet is not found.
        """
        positions = self._sphere_marker.get_positions()

        if positions["inlet"] is None:
            raise ValueError(
                "Inlet not found. Create a GREEN sphere or set config.inlet_prim_path."
            )
        if positions["outlet"] is None:
            raise ValueError(
                "Outlet not found. Create a RED sphere or set config.outlet_prim_path."
            )

        inlet = np.array(positions["inlet"], dtype=np.float64)
        outlet = np.array(positions["outlet"], dtype=np.float64)
        return inlet, outlet

    def extract_and_save(self, output_path: str = None):
        """
        Extract mesh + positions and save as a packed .npz file.
        This is the main entry point for the "Extract Scene Data" button.

        Parameters
        ----------
        output_path : str or None - uses config default if None

        Returns
        -------
        str - path to the saved .npz file
        """
        out = output_path or self.config.scene_data_path

        # Extract
        inlet_pos, outlet_pos = self.extract_positions()
        vertices, faces = self.extract_mesh()

        # Save
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez(
            out,
            vertices=vertices,
            faces=faces,
            inlet_pos=inlet_pos,
            outlet_pos=outlet_pos,
        )

        return out


# ============================================================================
# STANDALONE - paste into Omniverse Script Editor and run
# ============================================================================
if __name__ == "__main__":
    import carb

    cfg = CentrelineConfig()
    cfg.mesh_prim_path = "/World/OrganEnv/Colon/Colon"  # <- set your mesh path

    extractor = SceneExtractor(cfg)
    saved = extractor.extract_and_save()
    carb.log_info(f"Scene data saved: {saved}")
