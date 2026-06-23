"""
Centreline Tool - Sphere Marker
=================================
Create and manage inlet/outlet sphere markers in the Omniverse scene.

Can be used as:
  - Part of the centreline_tool package (import)
  - Standalone script in the Omniverse Script Editor (paste & run)

GUI button mapping:
  - "Create Inlet"  -> SphereMarker.create_inlet(position)
  - "Create Outlet" -> SphereMarker.create_outlet(position)
  - "Delete Markers" -> SphereMarker.delete_all()
"""

import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf

try:
    from .config import CentrelineConfig
except ImportError:
    from config import CentrelineConfig


class SphereMarker:
    """Manages inlet / outlet sphere markers in the USD scene."""

    def __init__(self, config: CentrelineConfig = None):
        self.config = config or CentrelineConfig()

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_stage(self):
        return omni.usd.get_context().get_stage()

    def _make_sphere(self, name: str, color: tuple, position: tuple, radius: float):
        """Create a coloured sphere prim and return its USD path string."""
        stage = self._get_stage()

        default_prim = stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            parent = default_prim.GetPath()
        else:
            parent = Sdf.Path("/World")

        sphere_path = parent.AppendChild(name)

        existing = stage.GetPrimAtPath(sphere_path)
        if existing and existing.IsValid():
            stage.RemovePrim(sphere_path)

        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.CreateRadiusAttr().Set(radius)

        # Position
        xform = UsdGeom.Xformable(sphere)
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))

        # Display colour
        prim = sphere.GetPrim()
        color_attr = prim.CreateAttribute(
            "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
        )
        color_attr.Set([Gf.Vec3f(*color)])
        color_attr.SetMetadata("interpolation", "constant")

        return str(sphere_path)

    # ── public API (one method = one button) ────────────────────────────

    def create_inlet(self, position=(0, 0, 0), name="Inlet_Marker", radius=None):
        """
        Create a GREEN sphere at *position* to mark an inlet.

        Parameters
        ----------
        position : tuple (x, y, z)
        name     : str - prim name
        radius   : float or None (uses config default)

        Returns
        -------
        str - USD prim path of the created sphere
        """
        r = radius if radius is not None else self.config.sphere_radius
        path = self._make_sphere(name, self.config.inlet_color, position, r)
        self.config.inlet_prim_path = path
        return path

    def create_outlet(self, position=(0, 0, 0), name="Outlet_Marker", radius=None):
        """
        Create a RED sphere at *position* to mark an outlet.

        Parameters
        ----------
        position : tuple (x, y, z)
        name     : str - prim name
        radius   : float or None (uses config default)

        Returns
        -------
        str - USD prim path of the created sphere
        """
        r = radius if radius is not None else self.config.sphere_radius
        path = self._make_sphere(name, self.config.outlet_color, position, r)
        self.config.outlet_prim_path = path
        return path

    def change_color(self, prim_path: str, marker_type: str) -> bool:
        """
        Change the colour of an existing sphere.

        Parameters
        ----------
        prim_path   : str - USD prim path
        marker_type : str - "inlet" (green) or "outlet" (red)

        Returns
        -------
        bool - True on success
        """
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False

        if marker_type.lower() == "inlet":
            color = Gf.Vec3f(*self.config.inlet_color)
        elif marker_type.lower() == "outlet":
            color = Gf.Vec3f(*self.config.outlet_color)
        else:
            return False

        attr = prim.GetAttribute("primvars:displayColor")
        if not attr.IsValid():
            attr = prim.CreateAttribute(
                "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
            )
            attr.SetMetadata("interpolation", "constant")
        attr.Set([color])
        return True

    def auto_detect(self):
        """
        Scan the scene for green / red spheres and return their world
        positions.  Updates config paths automatically.

        Returns
        -------
        dict - {"inlet": (x,y,z) or None, "outlet": (x,y,z) or None}
        """
        stage = self._get_stage()
        result = {"inlet": None, "outlet": None}

        for prim in stage.Traverse():
            if prim.GetTypeName() != "Sphere":
                continue
            color_attr = prim.GetAttribute("primvars:displayColor")
            if not color_attr.IsValid():
                continue
            colors = color_attr.Get()
            if not colors or len(colors) == 0:
                continue
            c = colors[0]

            if result["inlet"] is None and c[1] > 0.8 and c[0] < 0.2 and c[2] < 0.2:
                result["inlet"] = self._world_position(prim)
                self.config.inlet_prim_path = str(prim.GetPath())

            elif result["outlet"] is None and c[0] > 0.8 and c[1] < 0.2 and c[2] < 0.2:
                result["outlet"] = self._world_position(prim)
                self.config.outlet_prim_path = str(prim.GetPath())

        return result

    def get_positions(self):
        """
        Get world positions of the inlet and outlet markers.
        Uses explicit config paths if set, otherwise auto-detects.

        Returns
        -------
        dict - {"inlet": (x,y,z) or None, "outlet": (x,y,z) or None}
        """
        result = {"inlet": None, "outlet": None}

        if self.config.inlet_prim_path:
            stage = self._get_stage()
            prim = stage.GetPrimAtPath(self.config.inlet_prim_path)
            if prim.IsValid():
                result["inlet"] = self._world_position(prim)

        if self.config.outlet_prim_path:
            stage = self._get_stage()
            prim = stage.GetPrimAtPath(self.config.outlet_prim_path)
            if prim.IsValid():
                result["outlet"] = self._world_position(prim)

        # Fall back to auto-detect for any missing markers
        if result["inlet"] is None or result["outlet"] is None:
            detected = self.auto_detect()
            if result["inlet"] is None:
                result["inlet"] = detected["inlet"]
            if result["outlet"] is None:
                result["outlet"] = detected["outlet"]

        return result

    def list_markers(self):
        """
        List all coloured spheres in the scene.

        Returns
        -------
        list of dict - [{"path": str, "type": str, "position": tuple}, ...]
        """
        stage = self._get_stage()
        markers = []

        for prim in stage.Traverse():
            if prim.GetTypeName() != "Sphere":
                continue
            color_attr = prim.GetAttribute("primvars:displayColor")
            if not color_attr.IsValid():
                continue
            colors = color_attr.Get()
            if not colors or len(colors) == 0:
                continue
            c = colors[0]

            if c[1] > 0.8 and c[0] < 0.2 and c[2] < 0.2:
                marker_type = "inlet"
            elif c[0] > 0.8 and c[1] < 0.2 and c[2] < 0.2:
                marker_type = "outlet"
            else:
                continue

            pos = self._world_position(prim)
            markers.append({
                "path": str(prim.GetPath()),
                "type": marker_type,
                "position": pos,
            })

        return markers

    def delete_all(self):
        """Remove all inlet/outlet sphere markers from the scene."""
        stage = self._get_stage()
        markers = self.list_markers()
        for m in markers:
            stage.RemovePrim(m["path"])
        self.config.inlet_prim_path = ""
        self.config.outlet_prim_path = ""
        return len(markers)

    # ── internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _world_position(prim):
        xformable = UsdGeom.Xformable(prim)
        world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = world_xform.ExtractTranslation()
        return (t[0], t[1], t[2])


# ============================================================================
# STANDALONE - paste into Omniverse Script Editor and run
# ============================================================================
if __name__ == "__main__":
    import carb

    # ── Configuration ────────────────────────────────────────────────────
    INLET_POSITION  = (0, 0, 0)       # World position for inlet sphere
    OUTLET_POSITION = (100, 0, 0)     # World position for outlet sphere
    SPHERE_RADIUS   = 5.0
    # ─────────────────────────────────────────────────────────────────────

    marker = SphereMarker()
    marker.config.sphere_radius = SPHERE_RADIUS

    inlet_path = marker.create_inlet(position=INLET_POSITION)
    carb.log_info(f"Created inlet : {inlet_path}")

    outlet_path = marker.create_outlet(position=OUTLET_POSITION)
    carb.log_info(f"Created outlet: {outlet_path}")

    # List all markers in the scene
    for m in marker.list_markers():
        carb.log_info(f"  {m['type']:6s} @ {m['position']}  - {m['path']}")
