# EndoX [omni.endox]

An NVIDIA Omniverse extension for GPU-accelerated endoscopic perception simulation. EndoX provides a single GUI panel that drives the full synthetic-data workflow - from medical imaging input to multi-modal data capture - enabling researchers and engineers to generate large-scale training datasets for endoscopic AI.

## Pipeline Overview

The extension is organised into four sequential stages, each accessible from a collapsible section in the GUI:

### 1. 3-D Model Import

- **DICOM Pipeline**: Convert DICOM CT/MRI scans to 3-D organ meshes using MONAI Vista3D segmentation, with configurable organ labels, smoothing, decimation, and HU windowing. The resulting OBJ mesh is automatically converted to USD and added to the stage.
- **Direct OBJ Import**: Import an existing OBJ mesh file directly into the USD stage.

### 2. Centreline Extraction

- Place **inlet and outlet markers** interactively by clicking on the mesh surface in the viewport.
- **Extract scene data** (mesh geometry + marker positions) to a portable `.npz` file.
- **Compute anatomical centrelines** via an external VMTK subprocess (conda environment).
- **Visualise the motion path** as a USD BasisCurves prim with configurable curve type, basis, and width. This path can then be used to animate a capsule endoscope camera along the organ lumen.

### 3. AOV Data Capture

Capture GPU-accelerated multi-modal synthetic data using Omniverse Replicator and NVIDIA Warp:

| Modality | Format |
|----------|--------|
| RGB | PNG |
| Depth | 32-bit float NPY |
| Surface Normals (world space) | 16-bit float NPY |
| Surface Normals (camera space) | 16-bit float NPY |
| Optical Flow | 16-bit float NPY |
| Camera Pose | 4x4 extrinsic matrix NPY |
| Occlusion Map | Clipping-plane sweep PNG |

Supports configurable resolution, samples-per-pixel, write interval, frame count, and pause/stop controls.

### 4. Coverage Map

- Compute **per-vertex camera visibility** across all timeline frames using Warp GPU ray-casting.
- Apply **red/blue vertex colour overlays** (red = visible, blue = not visible) to visualise coverage.
- Toggle between coverage colours and photorealistic materials.
- Save and reload coverage data from `.npz` files.
