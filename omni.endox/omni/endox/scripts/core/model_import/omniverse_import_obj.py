"""
Omniverse Script Editor: convert OBJ to USD and add to the stage.

Set PRIM_PATH below. For OBJ_PATH: set it explicitly, or leave None to use the
path from the last dicom2mesh.py run (reads last_obj_output.txt in DICOM2MESH_DIR).
Requires Asset Converter extension (Extensions > Utility > Asset Converter).
"""

import asyncio
import os

# -----------------------------------------------------------------------------
# INPUT - set before running
# -----------------------------------------------------------------------------
# Folder containing dicom2mesh.py and last_obj_output.txt (written by dicom2mesh when run with -o)
DICOM2MESH_DIR = r""

# OBJ to convert: set a path, or None to use the path from last dicom2mesh.py run
OBJ_PATH = r""
PRIM_PATH = "/World/ImportedMesh"

# -----------------------------------------------------------------------------
# Implementation
# -----------------------------------------------------------------------------

async def convert_and_import(obj_path: str, prim_path: str) -> str:
    import omni.kit.asset_converter
    import omni.usd

    obj_path = os.path.abspath(obj_path)
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    base, _ = os.path.splitext(obj_path)
    usd_path = os.path.abspath(base + ".usd")

    def progress(current: int, total: int):
        if total > 0:
            print(f"  Converter: {current}/{total}")

    converter = omni.kit.asset_converter.get_instance()
    task = converter.create_converter_task(obj_path, usd_path, progress)
    success = await task.wait_until_finished()
    if not success:
        raise RuntimeError(f"Conversion failed: {task.get_error_message()}")

    if not os.path.isfile(usd_path):
        raise RuntimeError(f"Output USD not found: {usd_path}")

    stage = omni.usd.get_context().get_stage()
    if not stage:
        raise RuntimeError("No USD stage.")

    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)
    print("  Done. Mesh at", prim_path)
    return usd_path


def _resolve_obj_path():
    """Use OBJ_PATH if set, else path from last dicom2mesh.py run."""
    if OBJ_PATH and os.path.isfile(OBJ_PATH):
        return os.path.abspath(OBJ_PATH)
    last_file = os.path.join(DICOM2MESH_DIR, "last_obj_output.txt")
    if os.path.isfile(last_file):
        with open(last_file, "r") as f:
            path = f.read().strip()
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return OBJ_PATH  # may be None; caller will error


def main():
    print("OBJ -> USD and import")
    print("-" * 40)
    obj_path = _resolve_obj_path()
    if not obj_path or not os.path.isfile(obj_path):
        print("  Error: No OBJ path. Set OBJ_PATH or run dicom2mesh.py with -o <path> first.")
        return
    print("  OBJ:", obj_path)
    try:
        from omni.kit.async_engine import run_coroutine
        run_coroutine(convert_and_import(obj_path, PRIM_PATH))
        print("  Queued; mesh will appear at", PRIM_PATH)
    except ImportError:
        asyncio.run(convert_and_import(obj_path, PRIM_PATH))
        print("  Done. Mesh at", PRIM_PATH)
    except Exception as e:
        print("  Error:", e)


if __name__ == "__main__":
    main()
