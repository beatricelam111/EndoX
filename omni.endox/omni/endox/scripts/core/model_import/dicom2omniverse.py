"""
End-to-end DICOM -> OBJ, then (if in Omniverse) convert to USD and add to stage.

Runs in the same Python as dicom2mesh (needs dicom2nifti, nibabel, vtk, cupy, monai).
Calls dicom2mesh.run_pipeline() directly. If Omniverse is available, also runs
omniverse_import_obj.convert_and_import(); otherwise prints the OBJ path for manual import.
"""

import asyncio
import os
import tempfile
from typing import Any, Optional

import dicom2mesh
import omniverse_import_obj

# -----------------------------------------------------------------------------
# Default config - edit these, then call run()
# -----------------------------------------------------------------------------
CONFIG = {
    "bundle_path": r"",
    "dicom_path": r"",
    "prim_path": "/World/DicomMesh",
    "output_obj_path": None,
    "label_id": 19,
    "upsample": 1.0,
    "smoothing": 0.3,
    "sigma": 2.0,
    "hu_min": -500.0,
    "hu_max": 500.0,
    "isovalue": 0.0,
    "decimation": 0.1,
}


def run(**overrides: Any) -> None:
    """
    Run the pipeline using CONFIG, with optional keyword overrides.
    Example: run() or run(dicom_path=r"C:\other\dicom", label_id=12).
    """
    cfg = {**CONFIG, **overrides}
    pipeline = DICOM2Omniverse(
        bundle_path=cfg["bundle_path"],
        dicom_path=cfg["dicom_path"],
        prim_path=cfg["prim_path"],
        output_obj_path=cfg["output_obj_path"],
        label_id=cfg["label_id"],
        upsample=cfg["upsample"],
        smoothing=cfg["smoothing"],
        sigma=cfg["sigma"],
        hu_min=cfg["hu_min"],
        hu_max=cfg["hu_max"],
        isovalue=cfg["isovalue"],
        decimation=cfg["decimation"],
    )
    pipeline.run()


class DICOM2Omniverse:
    """
    End-to-end pipeline: call dicom2mesh.run_pipeline() (DICOM -> OBJ), then
    if running in Omniverse, convert OBJ to USD and add to the stage.
    """

    def __init__(
        self,
        bundle_path: str,
        dicom_path: str,
        prim_path: str = "/World/DicomMesh",
        output_obj_path: Optional[str] = None,
        label_id: int = 19,
        upsample: float = 1.0,
        smoothing: float = 0.3,
        sigma: float = 2.0,
        hu_min: float = -500.0,
        hu_max: float = 500.0,
        isovalue: float = 0.0,
        decimation: float = 0.1,
    ):
        self.bundle_path = os.path.abspath(bundle_path)
        self.dicom_path = os.path.abspath(dicom_path)
        self.prim_path = prim_path
        self.output_obj_path = output_obj_path
        self.label_id = label_id
        self.upsample = upsample
        self.smoothing = smoothing
        self.sigma = sigma
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.isovalue = isovalue
        self.decimation = decimation

        self._obj_path: Optional[str] = None
        self._cleanup_obj = False

    def run(self) -> None:
        """Run dicom2mesh.run_pipeline(), then (if in Omniverse) convert and import to stage."""
        print("DICOM -> OBJ (then to Omniverse if available)")
        print("-" * 50)

        if not os.path.isdir(self.bundle_path):
            print("  Error: Bundle path not found:", self.bundle_path)
            return
        if not os.path.isdir(self.dicom_path):
            print("  Error: DICOM path not found:", self.dicom_path)
            return

        if self.output_obj_path:
            obj_path = os.path.abspath(self.output_obj_path)
            self._cleanup_obj = False
        else:
            fd, obj_path = tempfile.mkstemp(suffix=".obj", prefix="dicom_mesh_")
            os.close(fd)
            self._cleanup_obj = True

        # Step 1: Call dicom2mesh.run_pipeline() directly
        print("[1/2] Running dicom2mesh.run_pipeline()...")
        try:
            obj_path = dicom2mesh.run_pipeline(
                vista3d_path=self.bundle_path,
                dicom_folder=self.dicom_path,
                output_obj=obj_path,
                label_id=self.label_id,
                upsample=self.upsample,
                smoothing_factor=self.smoothing,
                gaussian_sigma=self.sigma,
                hu_min=self.hu_min,
                hu_max=self.hu_max,
                isovalue=self.isovalue,
                decimation_reduction=self.decimation,
            )
        except Exception as e:
            print("  Error:", e)
            self._cleanup_temp_obj(obj_path)
            return

        if obj_path is None:
            print("  No OBJ path returned (run_pipeline was run without --output).")
            return
        self._obj_path = obj_path
        print("  OBJ:", self._obj_path)

        # Step 2: If in Omniverse, convert and add to stage
        print("[2/2] Converting OBJ to USD and adding to stage...")
        try:
            import omni.kit.asset_converter
            # We're in Omniverse; run the async convert and import
            async def _convert_and_import():
                try:
                    await omniverse_import_obj.convert_and_import(self._obj_path, self.prim_path)
                    self._cleanup_temp_obj(self._obj_path)
                    print("-" * 50)
                    print("Done. Mesh at", self.prim_path)
                except Exception as e:
                    print("  Error:", e)

            try:
                from omni.kit.async_engine import run_coroutine
                run_coroutine(_convert_and_import())
                print("  Conversion queued; mesh will appear at", self.prim_path)
            except ImportError:
                asyncio.run(_convert_and_import())
                self._cleanup_temp_obj(self._obj_path)
        except ImportError:
            print("  Not in Omniverse. OBJ path:", self._obj_path)
            print("  Import it in Omniverse with omniverse_import_obj (set OBJ_PATH and run).")
            self._cleanup_temp_obj(self._obj_path)

    def _cleanup_temp_obj(self, path: Optional[str]) -> None:
        if not self._cleanup_obj or not path:
            return
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass

    @property
    def obj_path(self) -> Optional[str]:
        """Path to the OBJ file after a successful run (None until run completes step 1)."""
        return self._obj_path


# -----------------------------------------------------------------------------
# When run as script, use CONFIG (edit CONFIG above or pass overrides)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run()
