"""
Centreline Tool - Pipeline
=============================
High-level orchestrator that ties all modules together.
Each public method maps to a GUI button.

GUI button mapping:
  ┌─────────────────────────┬──────────────────────────────────────────┐
  │ Button label            │ Method                                   │
  ├─────────────────────────┼──────────────────────────────────────────┤
  │ "Create Inlet Sphere"   │ pipeline.create_inlet(pos)               │
  │ "Create Outlet Sphere"  │ pipeline.create_outlet(pos)              │
  │ "Extract Scene Data"    │ pipeline.extract_scene_data()            │
  │ "Compute Centerline"    │ pipeline.compute_centerline()            │
  │ "Show Motion Path"      │ pipeline.show_motion_path()              │
  │ "Run All"               │ pipeline.run_all()                       │
  │ "Remove Motion Path"    │ pipeline.remove_motion_path()            │
  │ "Delete Markers"        │ pipeline.delete_markers()                │
  └─────────────────────────┴──────────────────────────────────────────┘
"""

import os
import numpy as np

try:
    from .config import CentrelineConfig
    from .sphere_marker import SphereMarker
    from .scene_extractor import SceneExtractor
    from .centerline_computer import CenterlineComputer
    from .motion_path import MotionPath
except ImportError:
    from config import CentrelineConfig
    from sphere_marker import SphereMarker
    from scene_extractor import SceneExtractor
    from centerline_computer import CenterlineComputer
    from motion_path import MotionPath


class CenterlinePipeline:
    """
    Top-level controller for the centreline extraction workflow.

    Usage (Script Editor)::

        from centreline_tool import CenterlinePipeline, CentrelineConfig

        cfg = CentrelineConfig()
        cfg.mesh_prim_path = "/World/OrganEnv/Colon/Colon"
        cfg.output_dir     = "/path/to/results"

        pipe = CenterlinePipeline(cfg)
        pipe.create_inlet(position=(-26, -194, 42))
        pipe.create_outlet(position=(-19, -213, -52))
        pipe.extract_scene_data()
        pipe.compute_centerline()   # VTK-only Voronoi if VMTK absent
        pipe.show_motion_path()

    Usage (GUI)::

        # Each button calls one method on a shared CenterlinePipeline instance.
        pipe = CenterlinePipeline(config)

        btn_inlet.on_click  = lambda pos: pipe.create_inlet(pos)
        btn_outlet.on_click = lambda pos: pipe.create_outlet(pos)
        btn_extract.on_click = pipe.extract_scene_data
        btn_compute.on_click = pipe.compute_centerline
        btn_show.on_click    = pipe.show_motion_path
    """

    def __init__(self, config: CentrelineConfig = None):
        self.config = config or CentrelineConfig()

        # Sub-modules share the same config instance
        self.sphere_marker = SphereMarker(self.config)
        self.scene_extractor = SceneExtractor(self.config)
        self.centerline_computer = CenterlineComputer(self.config)
        self.motion_path = MotionPath(self.config)

    # ── Sphere Markers ───────────────────────────────────────────────────

    def create_inlet(self, position=(0, 0, 0), name="Inlet_Marker", radius=None):
        """Button: Create Inlet Sphere."""
        return self.sphere_marker.create_inlet(position, name, radius)

    def create_outlet(self, position=(0, 0, 0), name="Outlet_Marker", radius=None):
        """Button: Create Outlet Sphere."""
        return self.sphere_marker.create_outlet(position, name, radius)

    def delete_markers(self):
        """Button: Delete all inlet/outlet markers."""
        return self.sphere_marker.delete_all()

    # ── Extract ──────────────────────────────────────────────────────────

    def extract_scene_data(self):
        """
        Button: Extract Scene Data.

        Reads mesh geometry and marker positions from the scene and saves
        them as a packed .npz file for external VMTK processing.

        Returns
        -------
        str - path to the saved .npz file
        """
        return self.scene_extractor.extract_and_save()

    # ── Compute ──────────────────────────────────────────────────────────

    def compute_centerline(self):
        """
        Button: Compute Centerline.

        Runs the full centreline pipeline on the .npz data:
          load -> watertight mesh -> extract -> orient -> clamp -> simplify -> save

        Extraction back-end is selected automatically:
          - VMTK ``vmtkCenterlines`` if importable
          - VTK-only Voronoi diagram (Antiga, 2003) otherwise

        This now works directly inside Omniverse Kit (no VMTK / conda
        needed).  The external subprocess path via
        ``compute_centerline_external()`` is still available as an
        alternative.

        Returns
        -------
        ndarray (M, 3) - simplified centreline
        """
        return self.centerline_computer.compute()

    # ── External Compute (subprocess) ────────────────────────────────────

    @staticmethod
    def _find_conda_python(env_name: str) -> str:
        """Auto-detect the Python executable for a conda environment."""
        home = os.path.expanduser("~")
        conda_exe = os.environ.get("CONDA_EXE", "")

        candidates = []

        # Derive from CONDA_EXE (most reliable)
        if conda_exe:
            conda_base = os.path.dirname(os.path.dirname(conda_exe))
            candidates.append(os.path.join(conda_base, "envs", env_name, "python.exe"))

        # Common Windows locations
        for base in ["miniconda3", "Miniconda3", "anaconda3", "Anaconda3"]:
            candidates.append(os.path.join(home, base, "envs", env_name, "python.exe"))

        candidates.append(os.path.join("C:\\", "ProgramData", "miniconda3", "envs", env_name, "python.exe"))

        for p in candidates:
            if os.path.isfile(p):
                return p
        return ""

    def _get_vmtk_python(self) -> str:
        """Return the VMTK Python path (explicit override or auto-detected).

        The conda_env field accepts either a bare environment name (e.g.
        ``vmtk``) or a full path to python.exe."""
        if self.config.vmtk_python_path:
            return self.config.vmtk_python_path

        env = self.config.conda_env
        if os.path.isfile(env):
            return env

        found = self._find_conda_python(env)
        if not found:
            raise FileNotFoundError(
                f"Could not find Python for conda env '{env}'.\n"
                f"Either:\n"
                f"  - Paste the full path to conda python.exe in the Conda Env field\n"
                f"  - Or ensure conda is installed in a standard location"
            )
        return found

    def compute_centerline_external(self):
        """
        Button: Compute Centerline (External).

        Launches centerline_computer.py in the conda VMTK environment via
        subprocess.  Auto-detects the conda Python from ``config.conda_env``
        or falls back to ``config.vmtk_python_path``.

        Output is streamed line-by-line to the Omniverse console.

        Returns
        -------
        int - subprocess return code (0 = success)
        """
        import subprocess

        python_exe = self._get_vmtk_python()
        script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "centerline_computer.py"
        )
        input_npz = self.config.scene_data_path
        output_npy = self.config.centerline_path

        cmd = [
            python_exe, script,
            "--input", input_npz,
            "--output", output_npy,
        ]

        print(f"[VMTK] Using Python: {python_exe}", flush=True)
        print(f"[VMTK] Input:  {input_npz}", flush=True)
        print(f"[VMTK] Output: {output_npy}", flush=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream output line-by-line
        for line in proc.stdout:
            print(f"[VMTK] {line}", end="", flush=True)

        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"VMTK subprocess failed (exit code {proc.returncode})"
            )

        return proc.returncode

    # ── Visualise ────────────────────────────────────────────────────────

    def show_motion_path(self, npy_path: str = None):
        """
        Button: Show Motion Path.

        Loads the .npy centreline and creates a BasisCurves prim in the scene.

        Parameters
        ----------
        npy_path : str or None - uses config default if None

        Returns
        -------
        UsdGeom.BasisCurves
        """
        return self.motion_path.create_from_npy(npy_path)

    def remove_motion_path(self):
        """Button: Remove Motion Path."""
        return self.motion_path.remove()

    # ── Run All ──────────────────────────────────────────────────────────

    def run_all(self):
        """
        Executes the full pipeline end-to-end:
          extract -> compute (VMTK or VTK-only) -> visualise
        """
        self.extract_scene_data()
        self.compute_centerline()
        self.show_motion_path()

    def run_all_external(self):
        """
        Button: Run All (Omniverse -> external VMTK -> Omniverse).

        Runs the complete pipeline from inside the Omniverse Script Editor:
          1. Extract scene data (.npz)
          2. Compute centreline via subprocess (conda VMTK)
          3. Visualise motion path in the scene

        Progress is printed to the Omniverse console.
        """
        print("=" * 50, flush=True)
        print("  Centreline Pipeline - Run All", flush=True)
        print("=" * 50, flush=True)

        # Step 1
        print("\n[Step 1/3] Extracting scene data ...", flush=True)
        saved = self.extract_scene_data()
        print(f"           Saved: {saved}", flush=True)

        # Step 2
        print("\n[Step 2/3] Computing centreline (external VMTK) ...", flush=True)
        self.compute_centerline_external()

        # Step 3
        print("\n[Step 3/3] Creating motion path ...", flush=True)
        curves = self.show_motion_path()
        print(f"           Created: {self.config.motion_path_full}", flush=True)

        print("\n" + "=" * 50, flush=True)
        print("  Pipeline complete!", flush=True)
        print("=" * 50, flush=True)

        return curves


# ============================================================================
# STANDALONE - paste into Omniverse Script Editor and run
# ============================================================================
if __name__ == "__main__":
    import carb

    cfg = CentrelineConfig()
    cfg.mesh_prim_path = "/World/OrganEnv/Colon/Colon"  # <- set your mesh path

    pipe = CenterlinePipeline(cfg)

    # ── Uncomment the steps you need ─────────────────────────────────────
    # pipe.extract_scene_data()
    # pipe.show_motion_path()
    # pipe.remove_motion_path()

    carb.log_info("Pipeline ready. Uncomment the steps you want to run.")
