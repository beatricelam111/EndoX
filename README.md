# EndoX

**GPU-accelerated endoscopic perception simulation in NVIDIA Omniverse**

EndoX is an Omniverse Kit extension that provides a unified GUI panel for generating large-scale, multi-modal synthetic endoscopy datasets with pixel-perfect ground truth. It covers the full pipeline from medical imaging input to data capture, eliminating the need for expensive and privacy-sensitive clinical data collection.

---
## Sample Dataset for Anonymous Submission

Sample dataset for Anonymous Submission to MICCAI 2026 

[Sample Dataset](https://zenodo.org/records/18792026?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc3MjE3MjU0MSwiZXhwIjoxNzkxNTkwMzk5fQ.eyJpZCI6ImYzNzg1YWNlLTNlMGYtNDVkNi04Yjg2LTliZDNiYTA0OTY3MiIsImRhdGEiOnt9LCJyYW5kb20iOiI0ZGJlNGUxOGU5ZTgzNmIyOThhYmZmNGQ0NDY0YjFjYSJ9.d_73DkwTAweNHgqvycmWH2Z4Kro3RnIE9yOV6JryFfJQ3ucNB7YexT0VR8GaMrsJIP_hoW0w10-wGbbjjcuqqg)


## Features

| Stage | Description |
|---|---|
| **3-D Model Import** | Convert DICOM CT/MRI scans to 3-D organ meshes via MONAI Vista3D segmentation, or import OBJ meshes directly into the USD stage. |
| **Centreline Extraction** | Interactive inlet/outlet marker placement, VMTK centreline computation, and BasisCurves motion-path visualisation for camera trajectory planning. |
| **AOV Data Capture** | GPU-accelerated multi-modal capture (RGB, depth, surface normals, optical flow, camera pose, occlusion) using Omniverse Replicator and NVIDIA Warp. |
| **Coverage Map** | Per-vertex camera visibility via Warp ray-casting with red/blue vertex-colour overlays and NPZ export/import. |

### Supported Modalities

| Modality | Format |
|---|---|
| RGB | PNG |
| Depth | 32-bit float NPY |
| Surface Normals (world) | 16-bit float NPY |
| Surface Normals (camera) | 16-bit float NPY |
| Optical Flow | 16-bit float NPY |
| Camera Pose | 4x4 extrinsic matrix NPY |
| Occlusion Map | Clipping-plane sweep PNG |

---

## Requirements

> **Tested on:** Windows 10, NVIDIA Omniverse Kit 106.5.7.

- **NVIDIA Omniverse** (Code or Kit-based app, Kit 106+)
- Omniverse extensions: `omni.ui`, `omni.usd`, `omni.timeline`, `omni.kit.viewport.utility`, `omni.replicator.core`, `omni.warp.core`, `omni.anim.motion_path.core`, `omni.anim.curve.core`, `omni.curve.creator`, `omni.curve.manipulator`
- **External Python packages** (bundled via `omni.pip_prebundle`, see below):

| Package | Used by | Purpose |
|---|---|---|
| `opencv-python` | AOV Capture | PNG encoding for depth, normals, optical flow |
| `vtk` | Centreline Extraction | VTK-only centreline computation (Voronoi diagram) |
| `scipy` | AOV Capture | Camera pose rotation matrices |
| `torch` | DICOM Pipeline | Required by MONAI Vista3D |
| `torchvision` | DICOM Pipeline | Required by MONAI |
| `torchaudio` | DICOM Pipeline | Required by MONAI |
| `pytorch-ignite` | DICOM Pipeline | Required by MONAI |
| `cupy-cuda12x` | DICOM Pipeline | GPU-accelerated DICOM processing |
| `monai[fire]` | DICOM Pipeline | Vista3D organ segmentation |
| `nibabel` | DICOM Pipeline | NIfTI file I/O |
| `dicom2nifti` | DICOM Pipeline | DICOM-to-NIfTI conversion |

- **Optional** (for VMTK centreline): conda environment with [VMTK](https://github.com/vmtk/vmtk) (the extension falls back to VTK-only extraction if VMTK is unavailable)

> **Note:** `numpy` and `Pillow` are already included in Kit via `omni.kit.pip_archive`. `warp` is provided by `omni.warp.core`.

---

## Prerequisites: Setting Up `omni.pip_prebundle`

Omniverse Kit uses its own embedded Python interpreter, so external pip packages must be pre-bundled into a companion extension. This section walks through creating a Kit app and an `omni.pip_prebundle` extension using the Kit App Template bundled in this repository.

### Step 1 - Create a Kit App

Extract the included `kit-app-template.zip` and navigate into the extracted directory:

```powershell
cd kit-app-template
```

Create a new application:

```powershell
.\repo.bat template new
```

Follow the prompts and select **Application**, then choose the desired template (e.g. Kit Base Editor).

### Step 2 - Create the `omni.pip_prebundle` extension

Run the template command again to create a new extension:

```powershell
.\repo.bat template new
```

This time select **Extension** and name it `omni.pip_prebundle`.

### Step 3 - Declare pip dependencies in `pip.toml`

A ready-to-use `pip.toml` is provided at [`tools/deps/pip.toml`](tools/deps/pip.toml). Copy it into your Kit project's `tools/deps/` directory (or merge it with your existing `pip.toml`).

It includes two dependency groups, both enabled by default:

- **Core** - `opencv-python`, `vtk`, `scipy` (required for AOV capture, centreline extraction, camera pose)
- **DICOM pipeline** - `torch`, `torchvision`, `torchaudio`, `pytorch-ignite`, `cupy-cuda12x`, `monai[fire]`, `nibabel`, `dicom2nifti` (required for in-process DICOM-to-mesh conversion via Vista3D)

> Adjust version numbers to match your Kit's Python version (Kit 106.x uses Python 3.10, Kit 107.x uses Python 3.11). You can check with `import sys; print(sys.version)` in the Omniverse Script Editor.

### Step 4 - Configure `premake5.lua`

In the root `premake5.lua`, ensure the `omni.pip_prebundle` extension includes a prebuild link so that the pip packages are available at runtime:

```lua
repo_build.prebuild_link {
    { "%{root}/_build/target-deps/pip_prebundle", ext.target_dir.."/pip_prebundle" },
}
```

### Step 5 - Add EndoX to the project

Copy or clone the `omni.endox` directory into the Kit project's extensions folder:

```
kit-app-template/source/extensions/omni.endox/
```

### Step 6 - Enable both extensions

Open your application `.kit` file (located in `source/apps/`) with your IDE and add the following lines to the `[dependencies]` section:

```toml
"omni.pip_prebundle" = {}
"omni.endox" = {}
```

### Step 7 - Build

Run the project build so that `repo_build` installs the pip packages and compiles all extensions:

```powershell
.\repo.bat build
```

Verify the packages appeared in `_build/target-deps/pip_prebundle/` (you should see folders like `cv2/`, `vtk/`, `scipy/`, etc.).

> If you later add or change packages in `pip.toml`, rebuild with the `-x` flag to force a clean reinstall:
> ```
> .\repo.bat build -x
> ```

### Step 8 - Launch

Start the Kit application in developer mode:

```powershell
.\repo.bat launch -d
```

The EndoX panel will appear in the Omniverse GUI.

---

## Quick Start

### Demo Scene

A pre-built demo scene is included at `omni/endox/demo/demo_scene.usd`. To get started quickly:

1. Open Omniverse Code (or your Kit-based app) with EndoX enabled.
2. Go to **File -> Open** and navigate to the extension directory:
   ```
   <exts_path>/omni.endox/omni/endox/demo/demo_scene.usd
   ```
3. The scene will load with a pre-configured organ mesh, camera, and materials ready for centreline extraction, AOV capture, and coverage mapping.

**Demo scene attribution:** The 3D organ mesh in the demo scene was reconstructed from one sample of CT colonography data provided by the CT COLONOGRAPHY collection [1] hosted on The Cancer Imaging Archive. The mesh surface texture was adopted from VR-Caps [2].

> [1] Smith, K., Clark, K., Bennett, W., Nolan, T., Kirby, J., Wolfsberger, M., Moulton, J., Vendt, B., Freymann, J.: Data from CT COLONOGRAPHY. The Cancer Imaging Archive (2015).
>
> [2] Incetan, K., Celik, I.O., Obeid, A., Gokceler, G.I., Ozyoruk, K.B., Almalioglu, Y., Chen, R.J., Mahmood, F., Gilbert, H., Durr, N.J., Turan, M.: VR-Caps: A virtual environment for capsule endoscopy. Medical Image Analysis **70**, 101990 (2021).

### 1. Import a 3-D Model

**Option A - DICOM pipeline:**
Provide a DICOM folder and a Vista3D model bundle path, configure segmentation settings (organ label, smoothing, decimation), then click **Run DICOM Pipeline**. The extension segments the scan, generates an OBJ mesh, converts it to USD, and adds it to the stage.
![import_dicom_mesh](docs/assets/import_dicom_mesh.gif)

**Option B - Direct OBJ import:**
Provide a path to an existing OBJ file and click **Import OBJ to Stage**.

### 2. Extract a Centreline

1. Click **Pick Inlet on Mesh** and click on the organ mesh surface to place the inlet marker.
2. Click **Pick Outlet on Mesh** to place the outlet marker.
3. Click **Run All (Extract -> VMTK -> Motion Path)** to extract scene data, compute the centreline, and display the motion path as a BasisCurves prim.

### 3. Capture Synthetic Data

1. Set the camera prim path, output directory, resolution, and frame count.
2. Select the desired modalities (RGB, Depth, Normals, Optical Flow, Camera Pose).
3. Click **Capture Selected Modalities** to begin GPU-accelerated capture.
![capture_modalities](docs/assets/capture_modalities.gif)

4. For occlusion maps, click **Capture Occlusion Only** (runs separately due to clipping-plane manipulation).
![capture_occlusion](docs/assets/capture_occlusion.gif)

### 4. Compute Coverage

1. Set the mesh and camera prim paths.
2. Click **Compute Coverage Map** to run Warp ray-casting across all timeline frames.
3. The mesh is coloured red (visible) / blue (not visible). Toggle **Show Materials** to switch back to photorealistic rendering.
4. Coverage data is saved as `.npz` and can be reloaded later.
![capture_coverage](docs/assets/capture_coverage.gif)

---

## Project Structure

```
omni.endox/
├── config/
│   └── extension.toml          # Extension metadata and dependencies
├── tools/
│   └── deps/
│       └── pip.toml            # pip_prebundle package list
├── docs/
│   ├── README.md               # Extension panel documentation
│   ├── Overview.md             # Architecture overview
│   └── CHANGELOG.md
├── premake5.lua                # Build configuration
└── omni/
    └── endox/
        ├── __init__.py
        ├── scripts/
        │   ├── extension.py    # Extension lifecycle (on_startup / on_shutdown)
        │   ├── window.py       # Main GUI panel (omni.ui)
        │   ├── actions.py      # Backend action dispatchers
        │   ├── widgets.py      # Reusable UI components
        │   ├── style.py        # UI style definitions
        │   └── core/
        │       ├── model_import/       # DICOM-to-mesh and OBJ import
        │       ├── centreline_extraction/  # VMTK / VTK-only centreline + motion path
        │       ├── aov_capture/        # Multi-modal synthetic data capture
        │       └── coverage_map/       # Warp GPU ray-casting coverage
        ├── tests/
        ├── demo/               # Demo scene assets
        └── data/               # Bundled USD assets and materials
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

