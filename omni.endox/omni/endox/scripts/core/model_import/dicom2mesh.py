"""
DICOM to 3D mesh pipeline: DICOM -> NIfTI -> MONAI Vista3D segmentation -> OBJ mesh.

Requires a separately installed Vista3D MONAI bundle (not included in this repo).
"""

import argparse
import json
import os
import subprocess
import sys

import cupy as cp
import dicom2nifti
import dicom2nifti.settings as settings
import nibabel as nib
import numpy as np
import vtk
from cupyx.scipy import ndimage as ndi
from vtk.util import numpy_support

DEFAULT_LABEL_PROMPT = 19  # Vista3D label ID (19 = small bowel); see bundle docs/labels.json for organ list


def convert_dicom_to_nifti(input_dir, output_folder, output_nifti="output.nii.gz"):
    """Convert DICOM directory to NIfTI and rename for standardized pipeline."""
    print("[1/4] Converting DICOM to NIfTI...")
    settings.disable_validate_slicecount()
    dicom2nifti.convert_directory(input_dir, output_folder, compression=True, reorient=True)
    out_file = None
    for fname in os.listdir(output_folder):
        if fname.endswith(".nii.gz"):
            src = os.path.join(output_folder, fname)
            dst = os.path.join(output_folder, output_nifti)
            os.replace(src, dst)
            out_file = dst
            break
    if not out_file:
        raise FileNotFoundError("No NIfTI file produced in output_folder.")
    print(f"      -> {out_file}")
    return out_file


def _find_python_exe():
    """Locate a usable ``python`` executable.

    Inside Omniverse Kit, ``sys.executable`` points to ``kit.exe``, which
    cannot be used to spawn Python subprocesses.  This helper searches
    known Kit-internal paths and the system PATH for a real interpreter.
    """
    import shutil

    candidates = [
        os.path.join(sys.prefix, "python.exe"),
        os.path.join(sys.prefix, "python"),
        os.path.join(sys.prefix, "bin", "python"),
        os.path.join(sys.prefix, "bin", "python3"),
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


def run_monai_inference(nifti_file, config_file, working_dir, label_id,
                        python_exe=None):
    """Run MONAI bundle inference (Vista3D) for a single organ.

    Tries to call ``monai.bundle.run()`` directly (in-process).
    Falls back to a subprocess if the direct call is unavailable.
    """
    print("[2/4] Running Vista3D segmentation...")
    nifti_file = os.path.abspath(nifti_file)
    input_list = [{"image": nifti_file, "label_prompt": [int(label_id)]}]
    input_json = json.dumps(input_list)

    # ── Try in-process first (works inside Omniverse Kit) ─────────
    try:
        from monai.bundle import run as monai_bundle_run

        prev_cwd = os.getcwd()
        os.chdir(working_dir)
        try:
            monai_bundle_run(
                config_file=config_file,
                input_dict=input_list,
            )
        finally:
            os.chdir(prev_cwd)

        print("      Vista3D segmentation complete.")
        return

    except ImportError:
        print("      monai.bundle not importable in-process, falling back to subprocess...")

    # ── Fallback: subprocess (for standalone Python environments) ──
    if python_exe is None:
        python_exe = _find_python_exe()
    command = [
        python_exe, "-m", "monai.bundle", "run",
        "--config_file", config_file,
        "--input_dict", input_json,
    ]
    print(f"      Python: {python_exe}")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    result = subprocess.run(command, cwd=working_dir, capture_output=True,
                            text=True, env=env)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"MONAI bundle run failed with code {result.returncode}")
    print("      Vista3D segmentation complete.")


def smooth_gaussian(binary_array, sigma=2):
    """Apply Gaussian smoothing on GPU and return result as NumPy array."""
    import cupy as cp
    from cupyx.scipy import ndimage as ndi
    gpu_array = cp.array(binary_array, dtype=cp.float32)
    vol = gpu_array - 0.5
    smoothed_gpu = ndi.gaussian_filter(vol, sigma=sigma)
    return cp.asnumpy(smoothed_gpu)


def convert_to_mesh(
    orig_nifti_file,
    vista3d_path,
    obj_file,
    upsample=1,
    smoothing_factor=0.3,
    gaussian_sigma=2,
    hu_min=-500,
    hu_max=500,
    isovalue=0.0,
    decimation_reduction=0.1,
    mask_file="eval/output/output_trans.nii.gz",
):
    """Extract mesh from NIfTI mask and save as OBJ using VTK."""
    print("[4/4] Extracting mesh...")
    print("      Loading NIfTI and segmentation mask...")
    orig_img = nib.load(orig_nifti_file)
    orig_data = orig_img.get_fdata()
    mask_path = mask_file if os.path.isabs(mask_file) else os.path.join(vista3d_path, mask_file)
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    print(f"      Building binary mask (HU range [{hu_min}, {hu_max}])...")
    mask = np.where(mask_data > 0, orig_data, 0)
    bin_mask = np.where((mask >= hu_min) & (mask <= hu_max), 1, 0).astype(np.float32)

    if upsample > 1:
        print(f"      Upsampling volume (factor={upsample})...")
        from cupyx.scipy.ndimage import zoom
        cp_vol = cp.array(bin_mask)
        upsampled = zoom(cp_vol, upsample, order=3)
        bin_mask = cp.asnumpy(upsampled)

    print(f"      Smoothing volume (Gaussian, sigma={gaussian_sigma})...")
    bin_mask = smooth_gaussian(bin_mask, sigma=gaussian_sigma)

    print("      Running Marching Cubes...")
    vtk_data_array = numpy_support.numpy_to_vtk(
        bin_mask.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT
    )
    image = vtk.vtkImageData()
    image.SetDimensions(bin_mask.shape)
    image.GetPointData().SetScalars(vtk_data_array)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.SetValue(0, isovalue)
    mc.Update()
    mesh = mc.GetOutput()
    print(f"      -> {mesh.GetNumberOfPoints()} vertices")

    print("      Extracting largest connected region...")
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(mesh)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    mesh = connectivity.GetOutput()
    print(f"      -> {mesh.GetNumberOfPoints()} vertices")
    if mesh.GetNumberOfPoints() == 0:
        print("      No points found. Skipping mesh export.")
        return

    print("      Smoothing mesh...")
    smoothing_filter = vtk.vtkWindowedSincPolyDataFilter()
    smoothing_filter.SetInputData(mesh)
    smoothing_filter.SetNumberOfIterations(int(20 + smoothing_factor * 40))
    smoothing_filter.SetPassBand(pow(10.0, -4.0 * smoothing_factor))
    smoothing_filter.BoundarySmoothingOff()
    smoothing_filter.FeatureEdgeSmoothingOff()
    smoothing_filter.NonManifoldSmoothingOn()
    smoothing_filter.NormalizeCoordinatesOn()
    smoothing_filter.Update()
    smoothed_mesh = smoothing_filter.GetOutput()
    print(f"      -> {smoothed_mesh.GetNumberOfPoints()} vertices")

    print(f"      Decimating mesh (target reduction={decimation_reduction})...")
    decimation = vtk.vtkQuadricDecimation()
    decimation.SetInputData(smoothed_mesh)
    decimation.SetTargetReduction(decimation_reduction)
    decimation.VolumePreservationOn()
    decimation.Update()
    final_mesh = decimation.GetOutput()
    print(f"      -> {final_mesh.GetNumberOfPoints()} vertices")

    print(f"      Writing OBJ: {obj_file}")
    obj_writer = vtk.vtkOBJWriter()
    obj_writer.SetInputData(final_mesh)
    obj_writer.SetFileName(obj_file)
    obj_writer.Write()
    print("      Mesh export complete.")


def run_pipeline(
    vista3d_path,
    dicom_folder,
    output_obj=None,
    config_file="configs/inference.json",
    label_id=None,
    nifti_name="output.nii.gz",
    mask_file="eval/output/output_trans.nii.gz",
    upsample=2,
    smoothing_factor=0.3,
    gaussian_sigma=2,
    hu_min=-500,
    hu_max=500,
    isovalue=0.0,
    decimation_reduction=0.1,
    python_exe=None,
):
    """Run full pipeline: DICOM -> NIfTI -> Vista3D inference -> OBJ mesh (one organ only)."""
    if label_id is None:
        label_id = DEFAULT_LABEL_PROMPT
    else:
        label_id = int(label_id) if isinstance(label_id, (int, float)) else int(label_id[0])

    vista3d_path = os.path.abspath(vista3d_path)
    dicom_folder = os.path.abspath(dicom_folder)
    if not os.path.isdir(vista3d_path):
        raise FileNotFoundError(f"Vista3D bundle path not found: {vista3d_path}")
    if not os.path.isdir(dicom_folder):
        raise FileNotFoundError(f"DICOM folder not found: {dicom_folder}")

    config_path = config_file if os.path.isabs(config_file) else os.path.join(vista3d_path, config_file)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print("Pipeline: DICOM -> NIfTI -> Vista3D -> OBJ (one organ)")
    print("-" * 50)
    print(f"      Label ID: {label_id} (see bundle docs/labels.json for organ names)")

    nifti_file = convert_dicom_to_nifti(dicom_folder, vista3d_path, output_nifti=nifti_name)
    run_monai_inference(nifti_file, config_file, vista3d_path, label_id,
                        python_exe=python_exe)

    pred_path = mask_file if os.path.isabs(mask_file) else os.path.join(vista3d_path, mask_file)
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not produced: {pred_path}")
    print("[3/4] Prediction file ready.")

    obj_path = None
    if output_obj:
        output_obj = os.path.abspath(output_obj)
        convert_to_mesh(
            nifti_file,
            vista3d_path,
            output_obj,
            upsample=upsample,
            smoothing_factor=smoothing_factor,
            gaussian_sigma=gaussian_sigma,
            hu_min=hu_min,
            hu_max=hu_max,
            isovalue=isovalue,
            decimation_reduction=decimation_reduction,
            mask_file=mask_file,
        )
        obj_path = output_obj
        # Write path for omniverse_import_obj.py (same dir as this script)
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(_script_dir, "last_obj_output.txt"), "w") as f:
            f.write(obj_path)
    else:
        print("[4/4] Skipped (no --output path given).")

    print("-" * 50)
    print("Done.")
    if obj_path is not None:
        print("OBJ:", obj_path)
    return obj_path


def main():
    parser = argparse.ArgumentParser(
        description="DICOM to 3D mesh: DICOM -> NIfTI -> Vista3D segmentation -> OBJ. "
                    "Requires Vista3D MONAI bundle installed separately."
    )
    parser.add_argument(
        "--bundle", "-b",
        required=True,
        help="Path to installed Vista3D MONAI bundle directory (e.g. .../vista3d)",
    )
    parser.add_argument(
        "--dicom", "-d",
        required=True,
        help="Path to folder containing DICOM series",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output OBJ file path (optional; if omitted, only NIfTI + segmentation run)",
    )
    parser.add_argument(
        "--config",
        default="configs/inference.json",
        help="Config file name or path relative to bundle (default: configs/inference.json)",
    )
    parser.add_argument(
        "--label-prompt",
        type=int,
        nargs=1,
        default=None,
        metavar="ID",
        help=f"Single organ label ID for Vista3D (e.g. 19=small bowel, 12=stomach). See bundle docs/labels.json. Default: {DEFAULT_LABEL_PROMPT}",
    )
    parser.add_argument(
        "--nifti-name",
        default="output.nii.gz",
        help="Output NIfTI filename written under bundle dir (default: output.nii.gz)",
    )
    parser.add_argument(
        "--mask-file",
        default="eval/output/output_trans.nii.gz",
        help="Path to Vista3D segmentation output, relative to bundle (default: eval/output/output_trans.nii.gz)",
    )
    parser.add_argument(
        "--upsample",
        type=float,
        default=1,
        help="Volume upsample factor before meshing (default: 1)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.3,
        help="Mesh smoothing factor (default: 0.3)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2,
        help="Gaussian sigma for volume smoothing before meshing (default: 2)",
    )
    parser.add_argument(
        "--hu-min",
        type=float,
        default=-500,
        help="Lower HU threshold for binary mask (default: -500)",
    )
    parser.add_argument(
        "--hu-max",
        type=float,
        default=500,
        help="Upper HU threshold for binary mask (default: 500)",
    )
    parser.add_argument(
        "--isovalue",
        type=float,
        default=0.0,
        help="Marching Cubes isosurface value (default: 0.0)",
    )
    parser.add_argument(
        "--decimation",
        type=float,
        default=0.1,
        help="Target reduction for mesh decimation, 0-1 (default: 0.1)",
    )
    args = parser.parse_args()

    run_pipeline(
        vista3d_path=args.bundle,
        dicom_folder=args.dicom,
        output_obj=args.output,
        config_file=args.config,
        label_id=args.label_prompt[0] if args.label_prompt is not None else None,
        nifti_name=args.nifti_name,
        mask_file=args.mask_file,
        upsample=args.upsample,
        smoothing_factor=args.smoothing,
        gaussian_sigma=args.sigma,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        isovalue=args.isovalue,
        decimation_reduction=args.decimation,
    )


if __name__ == "__main__":
    main()
