# EndoX: A GPU-accelerated Endoscopic Perception Simulation Framework


EndoX is an NVIDIA Omniverse Kit extension that provides a unified GUI panel for generating large-scale, multi-modal synthetic endoscopy datasets with pixel-perfect ground truth. It covers the full pipeline from medical imaging input to data capture, eliminating the need for expensive and privacy-sensitive clinical data collection.

<p align="center">
  <a href="https://beatricelam111.github.io/EndoX/"><img src="https://img.shields.io/badge/Project%20Page-EndoX-blue" alt="Project Page"/></a>
  <a href="https://huggingface.co/datasets/bealam111/EndoX_Dataset"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-yellow" alt="Dataset"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-green" alt="License"/></a>
</p>

## Features


| Stage                     | Description                                                                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **3-D Model Import**      | Convert DICOM CT/MRI scans to 3-D organ meshes via MONAI Vista3D segmentation, or import OBJ meshes directly into the USD stage.                    |
| **Centreline Extraction** | Interactive inlet/outlet marker placement, VMTK centreline computation, and BasisCurves motion-path visualisation for camera trajectory planning.   |
| **AOV Data Capture**      | GPU-accelerated multi-modal capture (RGB, depth, surface normals, optical flow, camera pose, occlusion) using Omniverse Replicator and NVIDIA Warp. |
| **Coverage Map**          | Per-vertex camera visibility via Warp ray-casting with red/blue vertex-colour overlays and NPZ export/import.                                       |


### Supported Modalities


| Modality                 | Format                   |
| ------------------------ | ------------------------ |
| RGB                      | PNG                      |
| Depth                    | 32-bit float NPY         |
| Surface Normals (world)  | 16-bit float NPY         |
| Surface Normals (camera) | 16-bit float NPY         |
| Optical Flow             | 16-bit float NPY         |
| Camera Pose              | 4x4 extrinsic matrix NPY |
| Occlusion Map            | Clipping-plane sweep PNG |


---

## Requirements

> **Tested on:** Windows 10, NVIDIA Omniverse Kit 106.5.7.

- **NVIDIA Omniverse** (Code or Kit-based app, Kit 106+)
- Omniverse extensions: `omni.ui`, `omni.usd`, `omni.timeline`, `omni.kit.viewport.utility`, `omni.replicator.core`, `omni.warp.core`, `omni.anim.motion_path.core`, `omni.anim.curve.core`, `omni.curve.creator`, `omni.curve.manipulator`
- **External Python packages** (bundled via `omni.pip_prebundle`, see below):


| Package          | Used by               | Purpose                                           |
| ---------------- | --------------------- | ------------------------------------------------- |
| `opencv-python`  | AOV Capture           | PNG encoding for depth, normals, optical flow     |
| `vtk`            | Centreline Extraction | VTK-only centreline computation (Voronoi diagram) |
| `scipy`          | AOV Capture           | Camera pose rotation matrices                     |
| `torch`          | DICOM Pipeline        | Required by MONAI Vista3D                         |
| `torchvision`    | DICOM Pipeline        | Required by MONAI                                 |
| `torchaudio`     | DICOM Pipeline        | Required by MONAI                                 |
| `pytorch-ignite` | DICOM Pipeline        | Required by MONAI                                 |
| `cupy-cuda12x`   | DICOM Pipeline        | GPU-accelerated DICOM processing                  |
| `monai[fire]`    | DICOM Pipeline        | Vista3D organ segmentation                        |
| `nibabel`        | DICOM Pipeline        | NIfTI file I/O                                    |
| `dicom2nifti`    | DICOM Pipeline        | DICOM-to-NIfTI conversion                         |


- **Optional** (for VMTK centreline): conda environment with [VMTK](https://github.com/vmtk/vmtk) (the extension falls back to VTK-only extraction if VMTK is unavailable)

> **Note:** `numpy` and `Pillow` are already included in Kit via `omni.kit.pip_archive`. `warp` is provided by `omni.warp.core`.

---

## Prerequisites: Setting Up `omni.pip_prebundle`

Omniverse Kit uses its own embedded Python interpreter, so external pip packages must be pre-bundled into a companion extension. This section walks through creating a Kit app and an `omni.pip_prebundle` extension using the Kit App Template bundled in this repository.

### Step 1 - Create a Kit App

Download or clone NVIDIA's official [Kit App Template](https://github.com/NVIDIA-Omniverse/kit-app-template), then navigate into the extracted or cloned directory:

```powershell
git clone https://github.com/NVIDIA-Omniverse/kit-app-template.git
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

A ready-to-use `pip.toml` is provided at `[tools/deps/pip.toml](tools/deps/pip.toml)`. Copy it into your Kit project's `tools/deps/` directory (or merge it with your existing `pip.toml`).

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
>
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
import_dicom_mesh

**Option B - Direct OBJ import:**
Provide a path to an existing OBJ file and click **Import OBJ to Stage**.

### 2. Extract a Centreline

1. Click **Pick Inlet on Mesh** and click on the organ mesh surface to place the inlet marker.
2. Click **Pick Outlet on Mesh** to place the outlet marker.
3. Click **Run All (Extract -> VMTK -> Motion Path)** to extract scene data, compute the centreline, and display the motion path as a BasisCurves prim.

### 3. Customise the Scene (Optional)

After importing the organ mesh and preparing the camera path, you can optionally edit the scene to match specific simulation scenarios. For example, you can author pathological mucosal textures, introduce polyp-bearing geometry, tune material properties, or adjust post-processing effects to simulate different endoscope appearances. See the [Scene Customization Tutorial](omni.endox/docs/CUSTOMIZE_SCENE.md) for details.

### 4. Capture Synthetic Data

1. Set the camera prim path, output directory, resolution, and frame count.
2. Select the desired modalities (RGB, Depth, Normals, Optical Flow, Camera Pose).
3. Click **Capture Selected Modalities** to begin GPU-accelerated capture.

capture_modalities

1. For occlusion maps, click **Capture Occlusion Only** (runs separately due to clipping-plane manipulation).

capture_occlusion

### 5. Compute Coverage

1. Set the mesh and camera prim paths.
2. Click **Compute Coverage Map** to run Warp ray-casting across all timeline frames.
3. The mesh is coloured red (visible) / blue (not visible). Toggle **Show Materials** to switch back to photorealistic rendering.
4. Coverage data is saved as `.npz` and can be reloaded later.

capture_coverage

---

## Performance Snapshot

The following table summarise EndoX scalability and speed on an NVIDIA RTX A6000 GPU (48 GB).


| Category                              | Metric                          | Result                               |
| ------------------------------------- | ------------------------------- | ------------------------------------ |
| Viewport frame rate                   | RTX / Path-Traced, capturing    | 23 / 32 FPS                          |
| Viewport frame rate                   | RTX / Path-Traced, static scene | 115 / 116 FPS                        |
| End-to-end generation                 | 4 modalities                    | 0.134 / 14.87 s per frame            |
| End-to-end generation                 | Occlusion                       | 5.76 s per frame                     |
| Warp acceleration for post-processing | GPU kernels vs CPU              | 2.4x-29.3x speedup across modalities |


---

## Project Structure

```
EndoX_camera_ready/
├── README.md                         # Main installation and quick-start guide
├── LICENSE
├── docs/                             # GitHub Pages project site
│   ├── index.html
│   ├── assets/                       # Project-page figures and GIFs
│   ├── static/                       # Website CSS, JS, images, videos, PDFs
│   └── tutorials/
│       └── pathological_texture_authoring/
│           ├── README.md
│           └── setup_assets.py
└── omni.endox/                       # Omniverse Kit extension package
    ├── config/
    │   └── extension.toml            # Extension metadata and dependencies
    ├── tools/
    │   └── deps/
    │       └── pip.toml              # pip_prebundle package list
    ├── docs/
    │   ├── README.md                 # Extension panel documentation
    │   ├── Overview.md               # Architecture overview
    │   ├── CHANGELOG.md
    │   ├── CUSTOMIZE_SCENE.md
    │   └── assets/
    ├── premake5.lua                  # Extension build configuration
    └── omni/
        └── endox/
            ├── __init__.py
            ├── scripts/
            │   ├── extension.py      # Extension lifecycle
            │   ├── window.py         # Main GUI panel (omni.ui)
            │   ├── actions.py        # Backend action dispatchers
            │   ├── widgets.py        # Reusable UI components
            │   ├── style.py          # UI style definitions
            │   └── core/
            │       ├── model_import/          # DICOM-to-mesh and OBJ import
            │       ├── centreline_extraction/ # VMTK / VTK centreline + motion path
            │       ├── aov_capture/           # Multi-modal synthetic data capture
            │       └── coverage_map/          # Warp ray-casting coverage
            ├── tests/
            ├── demo/                 # Demo scene assets
            └── data/                 # Bundled USD assets and materials
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.