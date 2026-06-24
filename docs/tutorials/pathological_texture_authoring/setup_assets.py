"""
Helper script to copy tutorial assets from the downloaded folder into the
assets/ subdirectory alongside the tutorial README.md.

Usage:
    python setup_assets.py

This script will:
  1. List all files in the source Pathological Texture folder
  2. Copy them into ./assets/ (creating the directory if needed)
  3. Print a mapping you can use to update README.md image references
"""

import os
import shutil

SOURCE_DIR = (
    r"C:\Users\nvaitc\Downloads"
    r"\Github_Manual_Authoring_Tutorial-20260624T075646Z-3-001"
    r"\Github_Manual_Authoring_Tutorial"
    r"\Pathological Texture"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"ERROR: Source directory not found:\n  {SOURCE_DIR}")
        return

    os.makedirs(ASSETS_DIR, exist_ok=True)

    print(f"Source : {SOURCE_DIR}")
    print(f"Target : {ASSETS_DIR}")
    print()

    copied = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for fname in files:
            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(src_path, SOURCE_DIR)
            dst_path = os.path.join(ASSETS_DIR, rel_path)

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied.append(rel_path)
            print(f"  Copied: {rel_path}")

    print(f"\n{len(copied)} file(s) copied to assets/\n")

    if copied:
        print("Files available for README.md image references:")
        for f in sorted(copied):
            md_path = f"assets/{f}".replace("\\", "/")
            print(f"  ![description]({md_path})")


if __name__ == "__main__":
    main()
