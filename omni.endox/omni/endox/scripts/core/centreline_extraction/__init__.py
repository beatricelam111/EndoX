"""
Centreline Tool
================
OOP toolkit for extracting and visualising organ centrelines in Omniverse.

Quick start (Omniverse Script Editor)::

    import sys
    for _m in [m for m in sys.modules if m.startswith("centreline_tool")]:
        del sys.modules[_m]

    from centreline_tool import CenterlinePipeline, CentrelineConfig

    cfg = CentrelineConfig()
    cfg.mesh_prim_path = "/World/OrganEnv/Colon/Colon"

    pipe = CenterlinePipeline(cfg)

    # Step 1 - Mark inlet / outlet in the scene
    pipe.create_inlet(position=(-26, -194, 42))
    pipe.create_outlet(position=(-19, -213, -52))

    # Step 2 - Extract scene data to .npz
    pipe.extract_scene_data()

    # Step 3 - Compute centreline (in conda VMTK env)
    pipe.compute_centerline()          # direct (if VMTK available)
    # pipe.compute_centerline_external()  # via subprocess

    # Step 4 - Visualise
    pipe.show_motion_path()

Modules
-------
- config              - CentrelineConfig (all settings)
- sphere_marker       - SphereMarker (inlet/outlet sphere management)
- scene_extractor     - SceneExtractor (USD mesh -> numpy)
- centerline_computer - CenterlineComputer (VMTK / VTK-only + post-processing)
- vtk_centerline      - VoronoiCenterline (VTK-only, no VMTK needed)
- motion_path         - MotionPath (BasisCurves visualisation)
- pipeline            - CenterlinePipeline (top-level orchestrator)
"""

try:
    from .config import CentrelineConfig
    from .sphere_marker import SphereMarker
    from .scene_extractor import SceneExtractor
    from .centerline_computer import CenterlineComputer
    from .vtk_centerline import VoronoiCenterline
    from .motion_path import MotionPath
    from .pipeline import CenterlinePipeline
except ImportError:
    from config import CentrelineConfig
    from sphere_marker import SphereMarker
    from scene_extractor import SceneExtractor
    from centerline_computer import CenterlineComputer
    from vtk_centerline import VoronoiCenterline
    from motion_path import MotionPath
    from pipeline import CenterlinePipeline

__all__ = [
    "CentrelineConfig",
    "SphereMarker",
    "SceneExtractor",
    "CenterlineComputer",
    "VoronoiCenterline",
    "MotionPath",
    "CenterlinePipeline",
]
