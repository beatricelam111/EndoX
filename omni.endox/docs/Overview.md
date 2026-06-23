# Overview

EndoX is an NVIDIA Omniverse extension for GPU-accelerated endoscopic perception simulation. It provides a single GUI panel to drive the full synthetic-data workflow - from DICOM-to-mesh conversion through centreline-guided camera trajectory creation to multi-modal data capture and coverage analysis.

## Motivation

Training robust endoscopic AI models requires large, diverse datasets with pixel-perfect ground truth. Acquiring such data from real clinical procedures is expensive, slow, and privacy-sensitive. EndoX bridges this gap by generating photorealistic synthetic endoscopy data inside NVIDIA Omniverse, with automatic ground-truth annotations (depth, normals, optical flow, camera pose, occlusion) at no additional labelling cost.

## Architecture

The extension is structured as four pipeline stages, each with its own GUI section:

1. **3-D Model Import** - DICOM CT/MRI scans are segmented with MONAI Vista3D and converted into textured USD meshes, or existing OBJ meshes can be imported directly.
2. **Centreline Extraction** - Interactive inlet/outlet marker placement on the mesh surface, followed by VMTK centreline computation and BasisCurves motion-path visualisation.
3. **AOV Data Capture** - Multi-modal capture (RGB, depth, surface normals, optical flow, camera pose, occlusion) using Omniverse Replicator annotators and NVIDIA Warp GPU kernels.
4. **Coverage Map** - Per-vertex camera visibility computation via Warp ray-casting, with red/blue vertex-colour overlays and NPZ export/import.
