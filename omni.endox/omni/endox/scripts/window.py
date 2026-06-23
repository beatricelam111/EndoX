"""
EndoX Pipeline - Main Extension Window

Sections
--------
1. 3-D Model Import
2. Centreline Extraction
3. AOV Data Capture
4. Coverage Map
"""

import asyncio
import traceback

import carb
import omni.ui as ui
import omni.ui.scene as sc
from omni.kit.viewport.utility import get_active_viewport_window

from .style import (
    main_style,
    SECTION_STYLE,
    TITLE_STYLE,
    SECTION_HEADER_STYLE,
    BUTTON_STYLE,
)
from .widgets import labeled_field, xyz_row, StatusLog
from . import actions


VISTA3D_LABELS = [
    (1, "Liver"), (2, "Kidney"), (3, "Spleen"), (4, "Pancreas"),
    (5, "Right kidney"), (6, "Aorta"), (7, "Inferior vena cava"),
    (8, "Right adrenal gland"), (9, "Left adrenal gland"),
    (10, "Gallbladder"), (11, "Esophagus"), (12, "Stomach"),
    (13, "Duodenum"), (14, "Left kidney"), (15, "Bladder"),
    (16, "Prostate or uterus"), (17, "Portal & splenic vein"),
    (18, "Rectum"), (19, "Small bowel"), (20, "Lung"), (21, "Bone"),
    (22, "Brain"), (23, "Lung tumor"), (24, "Pancreatic tumor"),
    (25, "Hepatic vessel"), (26, "Hepatic tumor"),
    (27, "Colon cancer primaries"), (28, "Left lung upper lobe"),
    (29, "Left lung lower lobe"), (30, "Right lung upper lobe"),
    (31, "Right lung middle lobe"), (32, "Right lung lower lobe"),
    (33, "Vertebrae L5"), (34, "Vertebrae L4"), (35, "Vertebrae L3"),
    (36, "Vertebrae L2"), (37, "Vertebrae L1"), (38, "Vertebrae T12"),
    (39, "Vertebrae T11"), (40, "Vertebrae T10"), (41, "Vertebrae T9"),
    (42, "Vertebrae T8"), (43, "Vertebrae T7"), (44, "Vertebrae T6"),
    (45, "Vertebrae T5"), (46, "Vertebrae T4"), (47, "Vertebrae T3"),
    (48, "Vertebrae T2"), (49, "Vertebrae T1"), (50, "Vertebrae C7"),
    (51, "Vertebrae C6"), (52, "Vertebrae C5"), (53, "Vertebrae C4"),
    (54, "Vertebrae C3"), (55, "Vertebrae C2"), (56, "Vertebrae C1"),
    (57, "Trachea"), (58, "Left iliac artery"), (59, "Right iliac artery"),
    (60, "Left iliac vena"), (61, "Right iliac vena"), (62, "Colon"),
    (63, "Left rib 1"), (64, "Left rib 2"), (65, "Left rib 3"),
    (66, "Left rib 4"), (67, "Left rib 5"), (68, "Left rib 6"),
    (69, "Left rib 7"), (70, "Left rib 8"), (71, "Left rib 9"),
    (72, "Left rib 10"), (73, "Left rib 11"), (74, "Left rib 12"),
    (75, "Right rib 1"), (76, "Right rib 2"), (77, "Right rib 3"),
    (78, "Right rib 4"), (79, "Right rib 5"), (80, "Right rib 6"),
    (81, "Right rib 7"), (82, "Right rib 8"), (83, "Right rib 9"),
    (84, "Right rib 10"), (85, "Right rib 11"), (86, "Right rib 12"),
    (87, "Left humerus"), (88, "Right humerus"),
    (89, "Left scapula"), (90, "Right scapula"),
    (91, "Left clavicula"), (92, "Right clavicula"),
    (93, "Left femur"), (94, "Right femur"),
    (95, "Left hip"), (96, "Right hip"), (97, "Sacrum"),
    (98, "Left gluteus maximus"), (99, "Right gluteus maximus"),
    (100, "Left gluteus medius"), (101, "Right gluteus medius"),
    (102, "Left gluteus minimus"), (103, "Right gluteus minimus"),
    (104, "Left autochthon"), (105, "Right autochthon"),
    (106, "Left iliopsoas"), (107, "Right iliopsoas"),
    (108, "Left atrial appendage"), (109, "Brachiocephalic trunk"),
    (110, "Left brachiocephalic vein"), (111, "Right brachiocephalic vein"),
    (112, "Left common carotid artery"), (113, "Right common carotid artery"),
    (114, "Costal cartilages"), (115, "Heart"),
    (116, "Left kidney cyst"), (117, "Right kidney cyst"),
    (118, "Prostate"), (119, "Pulmonary vein"), (120, "Skull"),
    (121, "Spinal cord"), (122, "Sternum"),
    (123, "Left subclavian artery"), (124, "Right subclavian artery"),
    (125, "Superior vena cava"), (126, "Thyroid gland"),
    (127, "Vertebrae S1"), (128, "Bone lesion"), (129, "Kidney mass"),
    (130, "Liver tumor"), (131, "Vertebrae L6"), (132, "Airway"),
]

_VISTA3D_DISPLAY = [f"{lid}: {name}" for lid, name in VISTA3D_LABELS]
_VISTA3D_DEFAULT_IDX = next(
    i for i, (lid, _) in enumerate(VISTA3D_LABELS) if lid == 19
)


class EndoxPipelineWindow(ui.Window):
    """Full GUI panel for the EndoX endoscopic-perception pipeline."""

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__(title, **kwargs)

        # ── 3-D Model Import - DICOM pipeline ────────────────────────
        self._dicom_path = ui.SimpleStringModel()
        self._dicom_bundle = ui.SimpleStringModel()
        self._dicom_bundle_dl_dir = ui.SimpleStringModel("C:/monai_bundles")
        self._dicom_output_obj = ui.SimpleStringModel()
        self._dicom_prim = ui.SimpleStringModel("/World/DicomMesh")
        self._dicom_label_combo = None  # set in _build_model_section
        self._dicom_upsample = ui.SimpleFloatModel(1.0)
        self._dicom_smoothing = ui.SimpleFloatModel(0.3)
        self._dicom_sigma = ui.SimpleFloatModel(2.0)
        self._dicom_hu_min = ui.SimpleFloatModel(-500.0)
        self._dicom_hu_max = ui.SimpleFloatModel(500.0)
        self._dicom_isovalue = ui.SimpleFloatModel(0.0)
        self._dicom_decimation = ui.SimpleFloatModel(0.1)

        # ── 3-D Model Import - direct OBJ import ─────────────────────
        self._obj_path = ui.SimpleStringModel()
        self._import_prim = ui.SimpleStringModel("/World/ImportedMesh")

        # ── Centreline Extraction ────────────────────────────────────
        self._cl_mesh = ui.SimpleStringModel()
        self._cl_inlet_x = ui.SimpleFloatModel(0.0)
        self._cl_inlet_y = ui.SimpleFloatModel(0.0)
        self._cl_inlet_z = ui.SimpleFloatModel(0.0)
        self._cl_outlet_x = ui.SimpleFloatModel(0.0)
        self._cl_outlet_y = ui.SimpleFloatModel(0.0)
        self._cl_outlet_z = ui.SimpleFloatModel(0.0)
        self._cl_output_dir = ui.SimpleStringModel()
        self._cl_sphere_radius = ui.SimpleFloatModel(5.0)
        self._cl_conda_env = ui.SimpleStringModel(
            "C:/Users/<your_username>/anaconda3/envs/vmtk"
        )
        self._cl_simplify = ui.SimpleBoolModel(True)
        self._cl_pct_keep = ui.SimpleFloatModel(0.10)
        self._cl_curve_type = ui.SimpleStringModel("cubic")
        self._cl_curve_basis = ui.SimpleStringModel("bspline")
        self._cl_curve_width = ui.SimpleFloatModel(2.0)

        # ── AOV Data Capture ─────────────────────────────────────────
        self._aov_camera = ui.SimpleStringModel("/World/CapsuleCam/Camera")
        self._aov_output = ui.SimpleStringModel("C:/output/aov_capture")
        self._aov_w = ui.SimpleIntModel(512)
        self._aov_h = ui.SimpleIntModel(512)
        self._aov_spp = ui.SimpleIntModel(512)
        self._aov_interval = ui.SimpleIntModel(5)
        self._aov_frames = ui.SimpleIntModel(0)
        self._mod_rgb = ui.SimpleBoolModel(True)
        self._mod_depth = ui.SimpleBoolModel(True)
        self._mod_nw = ui.SimpleBoolModel(True)
        self._mod_nc = ui.SimpleBoolModel(True)
        self._mod_of = ui.SimpleBoolModel(True)
        self._mod_cp = ui.SimpleBoolModel(True)
        self._occ_steps = ui.SimpleIntModel(100)
        self._occ_near = ui.SimpleFloatModel(1.0)
        self._occ_far = ui.SimpleFloatModel(100.0)
        self._occ_inc = ui.SimpleIntModel(1)
        self._occ_thresh = ui.SimpleFloatModel(0.05)
        self._occ_black = ui.SimpleFloatModel(0.02)

        # ── Coverage Map ─────────────────────────────────────────────
        self._cov_mesh = ui.SimpleStringModel()
        self._cov_camera = ui.SimpleStringModel("/World/CapsuleCam/Camera")
        self._cov_res = ui.SimpleIntModel(512)
        self._cov_save = ui.SimpleStringModel(
            "C:/output/coverage_map/coverage_vertices.npz"
        )
        self._cov_frame_save = ui.SimpleStringModel(
            "C:/output/coverage_map/coverage_vertices_per_frame.npz"
        )
        self._cov_nframes = ui.SimpleIntModel(0)
        self._cov_npz = ui.SimpleStringModel()

        # ── Internal state ───────────────────────────────────────────
        self._cl_pipeline = None
        self._aov_mgr = None
        self._cov_mgr = None
        self._cov_materials_hidden = False
        self._status_log: StatusLog | None = None

        # ── Viewport pick mode ───────────────────────────────────────
        self._pick_mode = None            # "inlet" | "outlet" | None
        self._pick_overlay_frame = None   # viewport overlay frame ref
        self._pick_scene_view = None      # SceneView used for the overlay

        self.frame.set_build_fn(self._build_fn)

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    def _collapsable(self, label: str, collapsed: bool = False):
        """Return a VStack wrapped in a CollapsableFrame."""
        v = None
        with ui.CollapsableFrame(
            label, height=0, collapsed=collapsed, style=SECTION_HEADER_STYLE
        ):
            with ui.ZStack():
                ui.Rectangle()
                v = ui.VStack()
        return v

    def _btn(self, label, fn, tooltip=""):
        ui.Button(
            label,
            height=28,
            clicked_fn=fn,
            tooltip=tooltip,
            style=BUTTON_STYLE,
        )

    def _log(self, msg: str):
        carb.log_info(msg)
        if self._status_log:
            self._status_log.log(msg)

    # ══════════════════════════════════════════════════════════════════
    # Build
    # ══════════════════════════════════════════════════════════════════

    def _build_fn(self):
        with self.frame:
            with ui.ScrollingFrame(style=main_style):
                with ui.VStack(height=0, spacing=6):
                    ui.Label("EndoX Pipeline", style=TITLE_STYLE)
                    ui.Line(height=2)

                    self._build_model_section()
                    self._build_centreline_section()
                    self._build_aov_section()
                    self._build_coverage_section()

                    ui.Line(height=2)
                    self._build_status_section()

        self._init_pick_overlay()

    # ──────────────────────────────────────────────────────────────────
    # Section 1 - 3-D Model Import
    # ──────────────────────────────────────────────────────────────────

    def _build_model_section(self):
        with self._collapsable("1.  3-D Model Import"):
            with ui.VStack(height=0, spacing=4, style=SECTION_STYLE):

                # ── 1a. DICOM -> Mesh -> Stage (primary) ───────────────
                with self._collapsable("DICOM -> Mesh -> Stage", collapsed=False):
                    with ui.VStack(height=0, spacing=4, style=SECTION_STYLE):
                        labeled_field("DICOM Folder", self._dicom_path)
                        labeled_field("Vista3D Bundle Path", self._dicom_bundle)

                        with self._collapsable("Download Vista3D Bundle", collapsed=True):
                            with ui.VStack(height=0, spacing=4, style=SECTION_STYLE):
                                labeled_field("Download Directory", self._dicom_bundle_dl_dir)
                                with ui.HStack(height=0):
                                    ui.Spacer(width=ui.Percent(10))
                                    self._btn(
                                        "Download Vista3D Bundle",
                                        self._on_download_bundle,
                                        tooltip="pip install monai first; downloads ~1 GB model weights",
                                    )
                                    ui.Spacer(width=ui.Percent(10))

                        labeled_field("Output OBJ Path", self._dicom_output_obj)
                        labeled_field("Stage Prim Path", self._dicom_prim)

                        with self._collapsable("Segmentation Settings", collapsed=True):
                            with ui.VStack(height=0, spacing=2):
                                with ui.HStack(height=24, style={"margin": 2}):
                                    ui.Label("Organ (Vista3D)", width=170)
                                    self._dicom_label_combo = ui.ComboBox(
                                        _VISTA3D_DEFAULT_IDX, *_VISTA3D_DISPLAY
                                    )
                                labeled_field("Upsample Factor", self._dicom_upsample, min=0.5, max=8.0, step=0.5)
                                labeled_field("Smoothing Factor", self._dicom_smoothing, min=0.0, max=1.0, step=0.05)
                                labeled_field("Gaussian Sigma", self._dicom_sigma, min=0.1, max=10.0, step=0.5)
                                labeled_field("HU Min", self._dicom_hu_min, min=-2000.0, max=2000.0, step=50.0)
                                labeled_field("HU Max", self._dicom_hu_max, min=-2000.0, max=2000.0, step=50.0)
                                labeled_field("Isovalue", self._dicom_isovalue, min=-1.0, max=1.0, step=0.05)
                                labeled_field("Decimation (0-1)", self._dicom_decimation, min=0.0, max=1.0, step=0.05)

                        ui.Spacer(height=4)
                        with ui.HStack(height=0):
                            ui.Spacer(width=ui.Percent(10))
                            self._btn(
                                "Run DICOM Pipeline",
                                self._on_run_dicom,
                                tooltip="DICOM -> NIfTI -> Vista3D segmentation -> OBJ -> USD -> Stage",
                            )
                            ui.Spacer(width=ui.Percent(10))

                ui.Spacer(height=4)

                # ── 1b. Direct OBJ Import (secondary) ────────────────
                with self._collapsable("Import Existing OBJ", collapsed=True):
                    with ui.VStack(height=0, spacing=4, style=SECTION_STYLE):
                        labeled_field("OBJ File Path", self._obj_path)
                        labeled_field("Stage Prim Path", self._import_prim)
                        ui.Spacer(height=4)
                        with ui.HStack(height=0):
                            ui.Spacer(width=ui.Percent(15))
                            self._btn(
                                "Import OBJ to Stage",
                                self._on_import_obj,
                                tooltip="Convert OBJ to USD and add to the current stage",
                            )
                            ui.Spacer(width=ui.Percent(15))

    # ──────────────────────────────────────────────────────────────────
    # Section 2 - Centreline Extraction
    # ──────────────────────────────────────────────────────────────────

    def _build_centreline_section(self):
        with self._collapsable("2.  Centreline Extraction"):
            with ui.VStack(height=0, spacing=4, style=SECTION_STYLE):
                labeled_field("Mesh Prim Path", self._cl_mesh)
                labeled_field("VMTK Conda Env Path", self._cl_conda_env)
                labeled_field("Centerline Output Dir", self._cl_output_dir)

                ui.Spacer(height=4)
                ui.Label("Step 1 - Place Inlet & Outlet Markers", style={"font_size": 14, "color": 0xFFCCCCCC})
                with ui.HStack(height=0, spacing=4):
                    self._btn(
                        "1a. Pick Inlet on Mesh",
                        lambda: self._enter_pick_mode("inlet"),
                        tooltip="Click on the mesh surface to place the inlet marker",
                    )
                    self._btn(
                        "1b. Pick Outlet on Mesh",
                        lambda: self._enter_pick_mode("outlet"),
                        tooltip="Click on the mesh surface to place the outlet marker",
                    )
                xyz_row("Inlet Position", self._cl_inlet_x, self._cl_inlet_y, self._cl_inlet_z)
                xyz_row("Outlet Position", self._cl_outlet_x, self._cl_outlet_y, self._cl_outlet_z)

                ui.Spacer(height=4)
                ui.Label("Step 2 - Compute & Visualise", style={"font_size": 14, "color": 0xFFCCCCCC})
                with ui.HStack(height=0):
                    ui.Spacer(width=ui.Percent(10))
                    self._btn(
                        "2. Run All (Extract -> VMTK -> Motion Path)",
                        self._on_cl_run_all,
                        tooltip="Extract scene data, compute centerline via VMTK, and show motion path",
                    )
                    ui.Spacer(width=ui.Percent(10))

                ui.Spacer(height=4)
                with ui.HStack(height=0, spacing=4):
                    self._btn("Delete Markers", self._on_cl_delete_markers)
                    self._btn("Remove Path", self._on_cl_remove_path)

                with self._collapsable("Manual Operation", collapsed=True):
                    with ui.VStack(height=0, spacing=4, style=SECTION_STYLE):
                        self._btn("M1. Extract Scene Data", self._on_cl_extract, tooltip="Mesh + markers -> .npz")
                        self._btn("M2. Compute Centerline", self._on_cl_compute, tooltip="Run VMTK via conda subprocess")
                        self._btn("M3. Show Motion Path", self._on_cl_show_path, tooltip="BasisCurves from .npy")

                with self._collapsable("Advanced Settings", collapsed=True):
                    with ui.VStack(height=0, spacing=2):
                        labeled_field("Sphere Radius", self._cl_sphere_radius, min=0.1, max=200.0)
                        labeled_field("Simplify Curve", self._cl_simplify)
                        labeled_field("Percent to Keep", self._cl_pct_keep, min=0.01, max=1.0, step=0.01)
                        labeled_field("Curve Type", self._cl_curve_type)
                        labeled_field("Curve Basis", self._cl_curve_basis)
                        labeled_field("Curve Width", self._cl_curve_width, min=0.1, max=50.0)

    # ──────────────────────────────────────────────────────────────────
    # Section 3 - AOV Data Capture
    # ──────────────────────────────────────────────────────────────────

    def _build_aov_section(self):
        with self._collapsable("3.  AOV Data Capture"):
            with ui.VStack(height=0, spacing=4, style=SECTION_STYLE):
                labeled_field("Camera Prim Path", self._aov_camera)
                labeled_field("Output Directory", self._aov_output)
                with ui.HStack(height=24, style={"margin": 2}):
                    ui.Label("Resolution", width=170)
                    ui.Label("W", width=14)
                    ui.IntDrag(model=self._aov_w, min=64, max=8192)
                    ui.Label("H", width=14)
                    ui.IntDrag(model=self._aov_h, min=64, max=8192)
                labeled_field("Samples Per Pixel", self._aov_spp, min=1, max=16384)
                labeled_field("Write Interval", self._aov_interval, min=1, max=100)
                labeled_field("Num Frames (0=all)", self._aov_frames, min=0, max=999999)

                with self._collapsable("Modalities", collapsed=False):
                    with ui.VStack(height=0, spacing=2):
                        labeled_field("RGB", self._mod_rgb)
                        labeled_field("Depth", self._mod_depth)
                        labeled_field("Normals (World)", self._mod_nw)
                        labeled_field("Normals (Camera)", self._mod_nc)
                        labeled_field("Optical Flow", self._mod_of)
                        labeled_field("Camera Pose", self._mod_cp)

                with self._collapsable("Occlusion Settings", collapsed=True):
                    with ui.VStack(height=0, spacing=2):
                        labeled_field("Clipping Steps", self._occ_steps, min=1, max=1000)
                        labeled_field("Near Clip", self._occ_near, min=0.01, max=1000.0, step=0.1)
                        labeled_field("Far Clip", self._occ_far, min=1.0, max=10000.0, step=1.0)
                        labeled_field("Near Increment", self._occ_inc, min=1, max=100)
                        labeled_field("Diff Threshold", self._occ_thresh, min=0.001, max=1.0, step=0.005)
                        labeled_field("Black Eps", self._occ_black, min=0.001, max=1.0, step=0.005)

                ui.Spacer(height=4)
                with ui.HStack(height=0, spacing=4):
                    self._btn(
                        "Capture Selected Modalities",
                        self._on_aov_capture,
                        tooltip="Capture all checked modalities (not occlusion)",
                    )
                with ui.HStack(height=0, spacing=4):
                    self._btn(
                        "Capture Occlusion Only",
                        self._on_aov_occlusion,
                        tooltip="Clipping-plane sweep (must run alone)",
                    )
                ui.Spacer(height=2)
                with ui.HStack(height=0, spacing=4):
                    self._aov_pause_btn = ui.Button(
                        "Pause",
                        height=28,
                        clicked_fn=self._on_aov_pause,
                        tooltip="Pause / Resume the running capture",
                        style=BUTTON_STYLE,
                        enabled=False,
                    )
                    self._aov_stop_btn = ui.Button(
                        "Stop",
                        height=28,
                        clicked_fn=self._on_aov_stop,
                        tooltip="Stop the running capture (keeps data captured so far)",
                        style=BUTTON_STYLE,
                        enabled=False,
                    )

    # ──────────────────────────────────────────────────────────────────
    # Section 4 - Coverage Map
    # ──────────────────────────────────────────────────────────────────

    def _build_coverage_section(self):
        with self._collapsable("4.  Coverage Map"):
            with ui.VStack(height=0, spacing=4, style=SECTION_STYLE):
                labeled_field("Mesh Prim Path", self._cov_mesh)
                labeled_field("Camera Prim Path", self._cov_camera)
                labeled_field("Raster Resolution", self._cov_res, min=64, max=4096)
                labeled_field("Save Path (.npz)", self._cov_save)
                labeled_field("Per-Frame Path (.npz)", self._cov_frame_save)
                labeled_field("Num Frames (0=all)", self._cov_nframes, min=0, max=999999)

                ui.Spacer(height=4)
                with ui.HStack(height=0, spacing=4):
                    self._btn(
                        "Compute Coverage Map",
                        self._on_cov_compute,
                        tooltip="Per-vertex visibility across the timeline",
                    )
                with ui.HStack(height=0, spacing=4):
                    self._cov_pause_btn = ui.Button(
                        "Pause",
                        height=28,
                        clicked_fn=self._on_cov_pause,
                        tooltip="Pause / Resume the coverage computation",
                        style=BUTTON_STYLE,
                        enabled=False,
                    )
                    self._cov_stop_btn = ui.Button(
                        "Stop",
                        height=28,
                        clicked_fn=self._on_cov_stop,
                        tooltip="Stop the coverage computation (saves data captured so far)",
                        style=BUTTON_STYLE,
                        enabled=False,
                    )

                ui.Spacer(height=8)
                ui.Label("Re-apply colours from saved data", style={"font_size": 14, "color": 0xFFAAAAAA})
                labeled_field("NPZ Path", self._cov_npz)
                with ui.HStack(height=0, spacing=4):
                    self._btn(
                        "Apply Colours from NPZ",
                        self._on_cov_recolor,
                        tooltip="Red = visible, Blue = not visible (uses Mesh Prim Path above)",
                    )
                    self._cov_mat_toggle_btn = ui.Button(
                        "Show Materials",
                        height=28,
                        clicked_fn=self._on_cov_toggle_materials,
                        tooltip="Toggle between coverage vertex colours and photorealistic materials",
                        style=BUTTON_STYLE,
                    )

    # ──────────────────────────────────────────────────────────────────
    # Status / Log
    # ──────────────────────────────────────────────────────────────────

    def _build_status_section(self):
        with self._collapsable("Status / Log", collapsed=False):
            self._status_log = StatusLog()

    # ══════════════════════════════════════════════════════════════════
    # Action handlers
    # ══════════════════════════════════════════════════════════════════

    # ── 1. Model Import ──────────────────────────────────────────────

    def _on_download_bundle(self):
        dl_dir = self._dicom_bundle_dl_dir.as_string.strip()
        if not dl_dir:
            self._log("Error: download directory is empty.")
            return
        self._log(f"Downloading Vista3D bundle to {dl_dir} (this may take a few minutes) ...")

        async def _do():
            try:
                bundle_path = await asyncio.get_event_loop().run_in_executor(
                    None, actions.download_vista3d_bundle, dl_dir,
                )
                self._dicom_bundle.set_value(bundle_path)
                self._log(f"Vista3D bundle ready: {bundle_path}")
            except ImportError:
                self._log(
                    "Error: monai is not installed. "
                    "Run:  pip install \"monai[fire,ignite]\" requests huggingface_hub"
                )
            except Exception as e:
                self._log(f"Bundle download failed: {e}")
                carb.log_error(traceback.format_exc())

        asyncio.ensure_future(_do())

    def _on_run_dicom(self):
        dicom = self._dicom_path.as_string.strip()
        bundle = self._dicom_bundle.as_string.strip()
        output = self._dicom_output_obj.as_string.strip()
        prim = self._dicom_prim.as_string.strip() or "/World/DicomMesh"

        if not dicom:
            self._log("Error: DICOM folder path is empty.")
            return
        if not bundle:
            self._log("Error: Vista3D bundle path is empty.")
            return
        if not output:
            self._log("Error: output OBJ path is empty.")
            return

        self._log(f"DICOM pipeline: {dicom} -> {prim} ...")

        async def _do():
            try:
                await actions.run_dicom_pipeline(
                    bundle_path=bundle,
                    dicom_path=dicom,
                    output_obj=output,
                    prim_path=prim,
                    label_id=VISTA3D_LABELS[
                        self._dicom_label_combo.model.get_item_value_model().as_int
                    ][0],
                    upsample=self._dicom_upsample.as_float,
                    smoothing=self._dicom_smoothing.as_float,
                    sigma=self._dicom_sigma.as_float,
                    hu_min=self._dicom_hu_min.as_float,
                    hu_max=self._dicom_hu_max.as_float,
                    isovalue=self._dicom_isovalue.as_float,
                    decimation=self._dicom_decimation.as_float,
                )
                self._log(f"DICOM pipeline complete - mesh at {prim}")
            except Exception as e:
                self._log(f"DICOM pipeline failed: {e}")
                carb.log_error(traceback.format_exc())

        asyncio.ensure_future(_do())

    def _on_import_obj(self):
        obj = self._obj_path.as_string.strip()
        prim = self._import_prim.as_string.strip()
        if not obj:
            self._log("Error: OBJ file path is empty.")
            return
        if not prim:
            prim = "/World/ImportedMesh"
        self._log(f"Importing {obj} -> {prim} ...")

        async def _do():
            try:
                await actions.import_obj_to_stage(obj, prim)
                self._log(f"Import complete - prim at {prim}")
            except Exception as e:
                self._log(f"Import failed: {e}")
                carb.log_error(traceback.format_exc())

        asyncio.ensure_future(_do())

    # ── 2. Centreline ────────────────────────────────────────────────

    def _resolve_mesh_prim(self, model=None) -> str:
        """Return the mesh prim path from a GUI field, falling back to
        ``/World/DicomMesh`` then ``/World/ImportedMesh``.  Updates the
        GUI field when a fallback is used.  Returns an empty string and
        logs a warning if no mesh can be found.

        Parameters
        ----------
        model : ui.SimpleStringModel | None
            The string model to read/write.  Defaults to ``self._cl_mesh``.
        """
        import omni.usd

        if model is None:
            model = self._cl_mesh

        path = model.as_string.strip()
        if path:
            return path

        stage = omni.usd.get_context().get_stage()
        for candidate in ("/World/DicomMesh", "/World/ImportedMesh"):
            prim = stage.GetPrimAtPath(candidate)
            if prim and prim.IsValid():
                model.set_value(candidate)
                self._log(f"Mesh prim auto-detected: {candidate}")
                return candidate

        carb.log_warn(
            "No mesh prim found at /World/DicomMesh or /World/ImportedMesh. "
            "Please provide the Mesh Prim Path manually."
        )
        self._log("Warning: no mesh prim found - please fill in the Mesh Prim Path field.")
        return ""

    def _ensure_cl_pipeline(self):
        """Create the pipeline once and keep it alive; sync GUI values into
        the existing config so that state set by previous operations (e.g.
        inlet/outlet prim paths) is preserved."""
        if self._cl_pipeline is None:
            cfg = actions.create_centreline_config(
                mesh_prim_path=self._resolve_mesh_prim(),
                output_dir=self._cl_output_dir.as_string,
                sphere_radius=self._cl_sphere_radius.as_float,
                conda_env=self._cl_conda_env.as_string,
                simplify=self._cl_simplify.as_bool,
                percent_to_keep=self._cl_pct_keep.as_float,
                curve_type=self._cl_curve_type.as_string,
                curve_basis=self._cl_curve_basis.as_string,
                curve_width=self._cl_curve_width.as_float,
            )
            self._cl_pipeline = actions.create_centreline_pipeline(cfg)
        else:
            cfg = self._cl_pipeline.config
            cfg.mesh_prim_path = self._resolve_mesh_prim()
            out = self._cl_output_dir.as_string
            if out:
                cfg.output_dir = out
            cfg.sphere_radius = self._cl_sphere_radius.as_float
            cfg.conda_env = self._cl_conda_env.as_string
            cfg.simplify = self._cl_simplify.as_bool
            cfg.percent_to_keep = self._cl_pct_keep.as_float
            cfg.curve_type = self._cl_curve_type.as_string
            cfg.curve_basis = self._cl_curve_basis.as_string
            cfg.curve_width = self._cl_curve_width.as_float
        return self._cl_pipeline

    # ── Viewport pick-on-mesh ────────────────────────────────────────

    def _init_pick_overlay(self):
        """Create a persistent SceneView overlay on the viewport so the
        ClickGesture is already initialised when the user enters pick mode.
        Called once from ``_build_fn``."""
        if self._pick_scene_view is not None:
            return
        vp_win = get_active_viewport_window()
        if vp_win is None:
            return

        self._pick_overlay_frame = vp_win.get_frame("endox_pick_overlay")
        with self._pick_overlay_frame:
            self._pick_scene_view = sc.SceneView(
                aspect_ratio_policy=sc.AspectRatioPolicy.STRETCH,
            )
            with self._pick_scene_view.scene:
                sc.Screen(
                    gestures=[
                        sc.ClickGesture(on_ended_fn=self._on_viewport_pick),
                    ]
                )
        vp_win.viewport_api.add_scene_view(self._pick_scene_view)

    def _enter_pick_mode(self, marker_type: str):
        """Activate viewport click-to-place for *marker_type* ("inlet"/"outlet")."""
        self._pick_mode = marker_type
        self._init_pick_overlay()
        if self._pick_scene_view is None:
            self._log("Error: no active viewport window found.")
            self._pick_mode = None
            return
        self._log(f"Pick mode ON - click on the mesh to place the {marker_type} marker.")

    def _on_viewport_pick(self, sender=None, *args):
        """Called by ClickGesture on every viewport click.  Ignored unless
        pick mode is active."""
        if self._pick_mode is None:
            return

        vp_win = get_active_viewport_window()
        if vp_win is None:
            self._exit_pick_mode()
            return

        viewport_api = vp_win.viewport_api

        mouse_ndc = None
        if sender is not None and hasattr(sender, "gesture_payload"):
            try:
                mouse_ndc = sender.gesture_payload.mouse
            except Exception:
                pass
        if mouse_ndc is None and self._pick_scene_view is not None:
            try:
                mouse_ndc = self._pick_scene_view.scene.gesture_payload.mouse
            except Exception:
                pass
        if mouse_ndc is None:
            self._log("Could not determine click position - try again.")
            return

        pixel, in_viewport = viewport_api.map_ndc_to_texture_pixel(mouse_ndc)
        if not in_viewport:
            self._log("Click was outside the viewport - try again.")
            return

        marker_type = self._pick_mode

        def _query_cb(path, pos, *_args):
            if not path:
                self._log("No surface hit - click directly on the mesh.")
                return
            self._on_pick_query_result(path, pos, marker_type)

        viewport_api.request_query(pixel, _query_cb)

    def _on_pick_query_result(self, hit_path: str, hit_pos, marker_type: str):
        """Place a marker at the ray-hit position and update the GUI fields."""
        pos = (float(hit_pos[0]), float(hit_pos[1]), float(hit_pos[2]))
        try:
            pipe = self._ensure_cl_pipeline()
            if marker_type == "inlet":
                path = pipe.create_inlet(position=pos)
                self._cl_inlet_x.set_value(pos[0])
                self._cl_inlet_y.set_value(pos[1])
                self._cl_inlet_z.set_value(pos[2])
            else:
                path = pipe.create_outlet(position=pos)
                self._cl_outlet_x.set_value(pos[0])
                self._cl_outlet_y.set_value(pos[1])
                self._cl_outlet_z.set_value(pos[2])
            self._log(
                f"{marker_type.capitalize()} placed at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) - {path}"
            )
        except Exception as e:
            self._log(f"Pick {marker_type} failed: {e}")
            carb.log_error(traceback.format_exc())
        finally:
            self._exit_pick_mode()

    def _exit_pick_mode(self):
        """Leave pick mode.  The overlay stays alive for reuse."""
        if self._pick_mode is not None:
            self._log("Pick mode OFF.")
            self._pick_mode = None

    def _on_cl_delete_markers(self):
        try:
            pipe = self._ensure_cl_pipeline()
            n = pipe.delete_markers()
            self._log(f"Deleted {n} marker(s)")
        except Exception as e:
            self._log(f"Delete markers failed: {e}")
            carb.log_error(traceback.format_exc())

    def _on_cl_extract(self):
        try:
            pipe = self._ensure_cl_pipeline()
            saved = pipe.extract_scene_data()
            self._log(f"Scene data saved: {saved}")

            positions = pipe.sphere_marker.get_positions()
            if positions["inlet"]:
                self._cl_inlet_x.set_value(positions["inlet"][0])
                self._cl_inlet_y.set_value(positions["inlet"][1])
                self._cl_inlet_z.set_value(positions["inlet"][2])
            if positions["outlet"]:
                self._cl_outlet_x.set_value(positions["outlet"][0])
                self._cl_outlet_y.set_value(positions["outlet"][1])
                self._cl_outlet_z.set_value(positions["outlet"][2])
        except Exception as e:
            self._log(f"Extract scene data failed: {e}")
            carb.log_error(traceback.format_exc())

    def _on_cl_compute(self):
        self._log("Computing centerline via external VMTK subprocess ...")
        try:
            pipe = self._ensure_cl_pipeline()
            rc = pipe.compute_centerline_external()
            self._log(f"Centerline computation finished (exit code {rc})")
        except Exception as e:
            self._log(f"Centerline computation failed: {e}")
            carb.log_error(traceback.format_exc())

    def _on_cl_show_path(self):
        try:
            pipe = self._ensure_cl_pipeline()
            pipe.show_motion_path()
            self._log(f"Motion path created: {pipe.config.motion_path_full}")
        except Exception as e:
            self._log(f"Show motion path failed: {e}")
            carb.log_error(traceback.format_exc())

    def _on_cl_remove_path(self):
        try:
            pipe = self._ensure_cl_pipeline()
            removed = pipe.remove_motion_path()
            self._log("Motion path removed" if removed else "No motion path to remove")
        except Exception as e:
            self._log(f"Remove motion path failed: {e}")
            carb.log_error(traceback.format_exc())

    def _on_cl_run_all(self):
        self._log("Running full centreline pipeline (extract -> VMTK -> visualise) ...")
        try:
            pipe = self._ensure_cl_pipeline()
            pipe.run_all_external()
            self._log("Full centreline pipeline complete")
        except Exception as e:
            self._log(f"Run all failed: {e}")
            carb.log_error(traceback.format_exc())

    # ── 3. AOV Capture ───────────────────────────────────────────────

    def _gather_modalities(self) -> list[str]:
        mods = []
        if self._mod_rgb.as_bool:
            mods.append("rgb")
        if self._mod_depth.as_bool:
            mods.append("depth")
        if self._mod_nw.as_bool:
            mods.append("normals_world")
        if self._mod_nc.as_bool:
            mods.append("normals_camera")
        if self._mod_of.as_bool:
            mods.append("optical_flow")
        if self._mod_cp.as_bool:
            mods.append("camera_pose")
        return mods

    def _aov_set_controls_enabled(self, running: bool):
        if self._aov_pause_btn:
            self._aov_pause_btn.enabled = running
            self._aov_pause_btn.text = "Pause"
        if self._aov_stop_btn:
            self._aov_stop_btn.enabled = running

    def _on_aov_capture(self):
        mods = self._gather_modalities()
        if not mods:
            self._log("Error: no modalities selected.")
            return
        self._log(f"Starting AOV capture: {mods}")

        async def _do():
            try:
                mgr = actions.create_aov_manager(
                    camera_prim_path=self._aov_camera.as_string,
                    output_dir=self._aov_output.as_string,
                    resolution=(self._aov_w.as_int, self._aov_h.as_int),
                    modalities=mods,
                    samples_per_pixel=self._aov_spp.as_int,
                    write_interval=self._aov_interval.as_int,
                    num_frames=self._aov_frames.as_int or None,
                )
                self._aov_mgr = mgr
                self._aov_set_controls_enabled(True)
                await mgr.capture_all()
                status = "stopped" if mgr.is_stopped else "complete"
                self._log(f"AOV capture {status}")
            except Exception as e:
                self._log(f"AOV capture failed: {e}")
                carb.log_error(traceback.format_exc())
            finally:
                self._aov_mgr = None
                self._aov_set_controls_enabled(False)

        asyncio.ensure_future(_do())

    def _on_aov_occlusion(self):
        self._log("Starting occlusion capture (clipping-plane sweep) ...")

        async def _do():
            try:
                mgr = actions.create_aov_manager(
                    camera_prim_path=self._aov_camera.as_string,
                    output_dir=self._aov_output.as_string,
                    resolution=(self._aov_w.as_int, self._aov_h.as_int),
                    modalities=["occlusion"],
                    samples_per_pixel=self._aov_spp.as_int,
                    write_interval=self._aov_interval.as_int,
                    num_frames=self._aov_frames.as_int or None,
                    num_clipping_steps=self._occ_steps.as_int,
                    near=self._occ_near.as_float,
                    far=self._occ_far.as_float,
                    near_increment=self._occ_inc.as_int,
                    diff_threshold=self._occ_thresh.as_float,
                    black_eps=self._occ_black.as_float,
                )
                self._aov_mgr = mgr
                self._aov_set_controls_enabled(True)
                await mgr.capture_all()
                status = "stopped" if mgr.is_stopped else "complete"
                self._log(f"Occlusion capture {status}")
            except Exception as e:
                self._log(f"Occlusion capture failed: {e}")
                carb.log_error(traceback.format_exc())
            finally:
                self._aov_mgr = None
                self._aov_set_controls_enabled(False)

        asyncio.ensure_future(_do())

    def _on_aov_pause(self):
        mgr = self._aov_mgr
        if mgr is None:
            return
        if mgr.is_paused:
            mgr.resume()
            self._aov_pause_btn.text = "Pause"
            self._log("Capture resumed")
        else:
            mgr.pause()
            self._aov_pause_btn.text = "Resume"
            self._log("Capture paused")

    def _on_aov_stop(self):
        mgr = self._aov_mgr
        if mgr is None:
            return
        mgr.stop()
        self._log("Capture stop requested - finishing current batch ...")

    # ── 4. Coverage Map ──────────────────────────────────────────────

    def _cov_set_controls_enabled(self, running: bool):
        if self._cov_pause_btn:
            self._cov_pause_btn.enabled = running
            self._cov_pause_btn.text = "Pause"
        if self._cov_stop_btn:
            self._cov_stop_btn.enabled = running

    def _on_cov_compute(self):
        mesh = self._resolve_mesh_prim(self._cov_mesh)
        cam = self._cov_camera.as_string.strip()
        if not mesh or not cam:
            self._log("Error: mesh and camera prim paths are required.")
            return
        self._log("Computing coverage map ...")

        async def _do():
            try:
                mgr = actions.create_coverage_map_computer(
                    mesh_prim_path=mesh,
                    camera_prim_path=cam,
                    resolution=self._cov_res.as_int,
                    save_path=self._cov_save.as_string,
                    per_frame_save_path=self._cov_frame_save.as_string,
                    num_frames=self._cov_nframes.as_int or None,
                )
                self._cov_mgr = mgr
                self._cov_set_controls_enabled(True)
                await mgr.compute()
                status = "stopped" if mgr.is_stopped else "complete"
                self._log(f"Coverage map {status}")
            except Exception as e:
                self._log(f"Coverage map failed: {e}")
                carb.log_error(traceback.format_exc())
            finally:
                self._cov_mgr = None
                self._cov_set_controls_enabled(False)

        asyncio.ensure_future(_do())

    def _on_cov_pause(self):
        mgr = self._cov_mgr
        if mgr is None:
            return
        if mgr.is_paused:
            mgr.resume()
            self._cov_pause_btn.text = "Pause"
            self._log("Coverage map resumed")
        else:
            mgr.pause()
            self._cov_pause_btn.text = "Resume"
            self._log("Coverage map paused")

    def _on_cov_stop(self):
        mgr = self._cov_mgr
        if mgr is None:
            return
        mgr.stop()
        self._log("Coverage map stop requested - saving data captured so far ...")

    def _on_cov_recolor(self):
        npz = self._cov_npz.as_string.strip()
        mesh = self._resolve_mesh_prim(self._cov_mesh)
        if not npz or not mesh:
            self._log("Error: NPZ path and mesh prim path are required.")
            return
        try:
            result = actions.apply_coverage_colors(npz, mesh)
            if result is not None:
                n_red, n_blue = result
                self._log(
                    f"Coverage colours applied to {mesh}: "
                    f"{n_red} red (visible to camera), "
                    f"{n_blue} blue (not visible). "
                    f"Materials hidden to show vertex colours."
                )
                self._cov_materials_hidden = True
                self._cov_mat_toggle_btn.text = "Show Materials"
            else:
                self._log("Apply colours failed - check the log for details.")
        except Exception as e:
            self._log(f"Apply colours failed: {e}")
            carb.log_error(traceback.format_exc())

    def _on_cov_toggle_materials(self):
        mesh = self._resolve_mesh_prim(self._cov_mesh)
        if not mesh:
            self._log("Error: mesh prim path is required.")
            return
        try:
            if self._cov_materials_hidden:
                actions.set_coverage_materials_visible(mesh, True)
                self._cov_materials_hidden = False
                self._cov_mat_toggle_btn.text = "Show Coverage Colours"
                self._log("Photorealistic materials restored")
            else:
                actions.set_coverage_materials_visible(mesh, False)
                self._cov_materials_hidden = True
                self._cov_mat_toggle_btn.text = "Show Materials"
                self._log("Coverage vertex colours shown "
                          "(red = visible, blue = not visible)")
        except Exception as e:
            self._log(f"Toggle materials failed: {e}")
            carb.log_error(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════
    # Teardown
    # ══════════════════════════════════════════════════════════════════

    def destroy(self):
        if self._aov_mgr is not None:
            self._aov_mgr.stop()
            self._aov_mgr = None
        if self._cov_mgr is not None:
            self._cov_mgr.stop()
            self._cov_mgr = None
        self._pick_mode = None
        if self._pick_scene_view is not None:
            vp_win = get_active_viewport_window()
            if vp_win is not None:
                try:
                    vp_win.viewport_api.remove_scene_view(self._pick_scene_view)
                except Exception:
                    pass
            self._pick_scene_view = None
        if self._pick_overlay_frame is not None:
            self._pick_overlay_frame.clear()
            self._pick_overlay_frame = None
        self._cl_pipeline = None
        self._status_log = None
        super().destroy()
