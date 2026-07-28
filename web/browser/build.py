"""
web/browser/build.py

Assemble a self-contained static bundle for the browser-only Nahual demo.

The browser demo needs four kinds of files served from one directory root:
the page assets (index.html, app.js, style.css, session_bootstrap.py), the real
nahual Python source it runs under Pyodide, and the trained model artifacts.
During development those are fetched from the repository's own ``nahual/`` and
``models/`` directories, but a static host wants a single self-contained folder.

This script copies exactly the runtime files into ``web/dist/`` so that folder
can be uploaded to any static host (Netlify, Cloudflare Pages, GitHub Pages,
Render static site, ...) or served locally. The nahual source is *copied*, not
duplicated by hand, so ``nahual/`` remains the single source of truth.

Usage (from anywhere)::

    python3 web/browser/build.py

Then deploy the printed ``web/dist`` directory, or serve it locally with::

    python3 -m http.server --directory web/dist 8000
"""

import shutil
from pathlib import Path

# Repository layout resolved relative to this file so the script works from any
# working directory.
BROWSER_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = BROWSER_DIRECTORY.parent.parent
NAHUAL_DIRECTORY = PROJECT_ROOT / "nahual"
MODELS_DIRECTORY = PROJECT_ROOT / "models"
DIST_DIRECTORY = PROJECT_ROOT / "web" / "dist"

# Page assets served at the bundle root.
PAGE_ASSETS = [
    "index.html",
    "app.js",
    "style.css",
    "session_bootstrap.py",
]

# The pure-Python nahual modules the browser runs under Pyodide. Deliberately
# excludes the desktop-only modules (hand_landmarker, visualization,
# gesture_collector, data_inspector) that import cv2/mediapipe and are never
# used server-less in the browser.
NAHUAL_RUNTIME_MODULES = [
    "__init__.py",
    "gesture_heuristics.py",
    "gesture_trainer.py",
    "realtime_session.py",
]

# Model artifacts: the two trained classifiers plus the MediaPipe hand model.
MODEL_ASSETS = [
    "gesture_classifier.pkl",
    "dynamic_gesture_classifier.pkl",
    "hand_landmarker.task",
]


def build() -> None:
    """Copy all runtime files into a fresh web/dist/ bundle."""
    if DIST_DIRECTORY.exists():
        shutil.rmtree(DIST_DIRECTORY)
    (DIST_DIRECTORY / "nahual").mkdir(parents=True)
    (DIST_DIRECTORY / "models").mkdir(parents=True)

    for asset in PAGE_ASSETS:
        shutil.copy2(BROWSER_DIRECTORY / asset, DIST_DIRECTORY / asset)

    for module_name in NAHUAL_RUNTIME_MODULES:
        shutil.copy2(
            NAHUAL_DIRECTORY / module_name, DIST_DIRECTORY / "nahual" / module_name
        )

    for model_name in MODEL_ASSETS:
        shutil.copy2(
            MODELS_DIRECTORY / model_name, DIST_DIRECTORY / "models" / model_name
        )

    total_bytes = sum(
        path.stat().st_size for path in DIST_DIRECTORY.rglob("*") if path.is_file()
    )
    print(f"Built {DIST_DIRECTORY} ({total_bytes / 1_000_000:.1f} MB).")
    print("Serve locally:  python3 -m http.server --directory web/dist 8000")
    print("Then open:      http://localhost:8000")


if __name__ == "__main__":
    build()
