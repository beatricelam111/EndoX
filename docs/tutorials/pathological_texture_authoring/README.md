# Authoring Pathological Textures in NVIDIA Omniverse

This tutorial walks through the process of creating and applying pathological mucosal textures to GI organ meshes in NVIDIA Omniverse for synthetic endoscopy data generation with EndoX.

**What you will learn:**

- Prepare normal mucosal textures and pathological pattern images
- Overlay pathological patterns onto normal textures to create composite pathological textures
- Apply pathological textures to organ meshes in Omniverse via the Stage tree or texture painting
- Run the EndoX AOV Capture extension for multi-modal synthetic data generation

---

## Prerequisites

- NVIDIA Omniverse (Code or Kit-based app, Kit 106+) with EndoX enabled
- A GI organ mesh imported into the Omniverse stage (see the [main README](../../../README.md) for DICOM-to-mesh or OBJ import instructions)
- An image editor capable of alpha-compositing (e.g. GIMP, Photoshop, or a Python script with Pillow)

---

## Step 1: Prepare the Source Textures

Before authoring a pathological texture, you need two categories of source images:

### 1.1 Normal Mucosal Texture

The **normal mucosal texture** represents the healthy appearance of the GI organ lining. This is a seamless or semi-seamless diffuse texture image that will serve as the base layer.

<p align="center">
  <img src="assets/normal_mucosal_texture.png" width="480" alt="Normal mucosal texture of the GI organ"/>
</p>
<p align="center"><em>Normal mucosal texture of the GI organ surface.</em></p>

> **Tip:** You can source normal mucosal textures from existing endoscopy simulation datasets (e.g. VR-Caps) or photograph real tissue samples under controlled lighting. Ensure the texture is tileable if the organ mesh UV layout requires repeating.

### 1.2 Pathological Patterns

Pathological patterns are overlay images that represent specific disease appearances. Each pattern is prepared as a separate image (ideally with transparency) so it can be composited onto the normal texture.

**Ulcerative Colitis Pattern**

<p align="center">
  <img src="assets/ulcerative_colitis_pattern.png" width="480" alt="Ulcerative colitis pattern"/>
</p>
<p align="center"><em>Ulcerative colitis pattern showing mucosal inflammation and ulceration.</em></p>

**Bleeding Pattern**

<p align="center">
  <img src="assets/bleeding_pattern.png" width="480" alt="Bleeding pattern"/>
</p>
<p align="center"><em>Bleeding pattern representing active hemorrhagic regions on the mucosal surface.</em></p>

> **Note:** You can create additional pathological patterns (polyps, erosions, tumours, etc.) following the same preparation approach. Each pattern should be isolated on a transparent or solid background for clean compositing.

---

## Step 2: Overlay Patterns to Form Pathological Textures

Combine the normal mucosal texture (base layer) with one or more pathological patterns (overlay layers) to produce the final **pathological texture** that will be applied to the organ mesh.

### Compositing Workflow

1. Open the **normal mucosal texture** as the base layer in your image editor.
2. Import the **pathological pattern** (e.g. ulcerative colitis) as a new layer on top.
3. Adjust the position, scale, and opacity of the pattern layer to achieve the desired disease coverage and severity.
4. Flatten / export the composite as a single image (PNG or JPEG).

<p align="center">
  <img src="assets/overlay_texture_composition.png" width="720" alt="Overlay composition of pathological texture"/>
</p>
<p align="center"><em>The pathological pattern is overlaid onto the normal mucosal texture to form the final pathological texture.</em></p>

**Example: Composited Results**

| Normal Mucosal Texture | + Ulcerative Colitis Pattern | = Pathological Texture |
|:---:|:---:|:---:|
| ![base](assets/normal_mucosal_texture.png) | ![pattern](assets/ulcerative_colitis_pattern.png) | ![result](assets/pathological_texture_uc.png) |

| Normal Mucosal Texture | + Bleeding Pattern | = Pathological Texture |
|:---:|:---:|:---:|
| ![base](assets/normal_mucosal_texture.png) | ![pattern](assets/bleeding_pattern.png) | ![result](assets/pathological_texture_bleeding.png) |

> **Programmatic alternative:** You can also script the compositing step with Python and Pillow:
> ```python
> from PIL import Image
>
> base = Image.open("normal_mucosal_texture.png").convert("RGBA")
> pattern = Image.open("ulcerative_colitis_pattern.png").convert("RGBA")
> pattern = pattern.resize(base.size)
> composite = Image.alpha_composite(base, pattern)
> composite.save("pathological_texture_uc.png")
> ```

---

## Step 3: Apply the Pathological Texture in Omniverse

There are two approaches to apply the authored pathological texture to the organ mesh in Omniverse:

### Option A: Link Texture via the Stage Tree (Material Property Editor)

This method replaces the material's diffuse texture by navigating the Omniverse Stage hierarchy.

1. **Open the Stage panel** and expand the organ prim hierarchy to locate the material assigned to the mesh.

<p align="center">
  <img src="assets/folder_tree_navigate.png" width="720" alt="Navigating the Omniverse Stage tree to find the organ material"/>
</p>
<p align="center"><em>Navigate the folder/stage tree in Omniverse to locate the organ mesh and its assigned material.</em></p>

2. **Select the material prim** (e.g. `Looks/MucosalMaterial`) and open the **Property Editor** panel.

3. **Locate the diffuse texture input** (typically `inputs:diffuse_texture` or `inputs:albedo_tex` depending on the MDL shader) and click the file picker button.

4. **Browse to your pathological texture** image file and select it. The viewport will update to show the new texture applied to the organ mesh.

<p align="center">
  <img src="assets/link_texture_property.png" width="720" alt="Linking the pathological texture in the Property Editor"/>
</p>
<p align="center"><em>Edit the material's diffuse texture input to link the authored pathological texture file.</em></p>

### Option B: Texture Painting (Interactive)

For more localised control, Omniverse supports **texture painting** directly on the mesh surface. This allows you to paint pathological regions at specific locations rather than replacing the entire texture map.

1. Enable the texture painting tool from the Omniverse toolbar.
2. Select the pathological pattern as the brush texture/stamp.
3. Paint directly onto the organ mesh surface in the viewport. Adjust brush size, opacity, and falloff to control the appearance.

<p align="center">
  <img src="assets/texture_painting.gif" width="720" alt="Texture painting pathological patterns onto the organ mesh"/>
</p>
<p align="center"><em>Interactive texture painting allows precise placement of pathological patterns on the organ surface.</em></p>

> **When to use which approach:**
> - **Option A (Stage tree):** Best for applying a uniform pathological texture across the entire organ surface. Faster setup for generating large-scale datasets with consistent disease appearance.
> - **Option B (Texture painting):** Best for creating localised pathology (e.g. a bleeding region in a specific area) or combining multiple disease patterns at different locations on the same organ.

---

## Step 4: Run AOV Capture for Synthetic Data Generation

With the pathological texture applied to the organ mesh, the scene is ready for multi-modal synthetic data capture using the EndoX AOV Capture extension.

### Capture Workflow

1. Open the **EndoX** panel in Omniverse.
2. Ensure the **camera prim path** is set to the virtual endoscope camera, and the **output directory**, **resolution**, and **frame count** are configured.
3. Select the desired output modalities: **RGB**, **Depth**, **Surface Normals** (world/camera), **Optical Flow**, **Camera Pose**, and **Occlusion**.
4. Click **Capture Selected Modalities** to begin GPU-accelerated rendering.

The capture produces pixel-perfect ground-truth annotations alongside the RGB frames showing the pathological texture.

### Example Results

<p align="center">
  <img src="assets/aov_result_rgb.png" width="720" alt="RGB capture result with pathological texture"/>
</p>
<p align="center"><em>RGB frame captured from the virtual endoscope traversing the organ with pathological texture applied.</em></p>

<p align="center">
  <img src="assets/aov_result_multimodal.png" width="720" alt="Multi-modal AOV capture results"/>
</p>
<p align="center"><em>Multi-modal AOV capture results: RGB, depth, surface normals, optical flow, and occlusion maps are generated simultaneously with the pathological texture appearance.</em></p>

| RGB | Depth | Surface Normal | Occlusion |
|:---:|:---:|:---:|:---:|
| ![rgb](assets/aov_rgb.png) | ![depth](assets/aov_depth.png) | ![normal](assets/aov_normal.png) | ![occlusion](assets/aov_occlusion.png) |

<p align="center"><em>All ground-truth modalities remain geometrically accurate regardless of the surface texture applied. The pathological appearance affects only the RGB output, while depth, normals, optical flow, and occlusion annotations preserve exact correspondence to the underlying 3D geometry.</em></p>

---

## Summary

| Step | Action | Key Output |
|------|--------|------------|
| 1 | Prepare normal mucosal texture and pathological pattern images | Source texture files |
| 2 | Composite patterns onto the normal texture | Final pathological texture image |
| 3 | Link the texture in Omniverse (Stage tree or texture painting) | Organ mesh with pathological appearance |
| 4 | Run EndoX AOV Capture | Multi-modal synthetic dataset with pathological texture |

This workflow enables scalable generation of synthetic endoscopy datasets with diverse pathological appearances while maintaining pixel-perfect geometric ground truth. By varying the pathological patterns, overlay positions, and severity levels, you can produce large, diverse training datasets for downstream tasks such as disease detection, depth estimation, and 3D reconstruction.

---

## References

- EndoX: [Project Page](https://beatricelam111.github.io/EndoX/) | [GitHub](https://github.com/beatricelam111/EndoX)
- VR-Caps texture source: Incetan, K. et al. *VR-Caps: A virtual environment for capsule endoscopy.* Medical Image Analysis **70**, 101990 (2021).
- NVIDIA Omniverse: [Documentation](https://docs.omniverse.nvidia.com/)
