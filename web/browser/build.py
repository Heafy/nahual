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

Every supported sign language is bundled: the classifiers keep their
``models/<language>/`` layout inside the bundle so the browser can fetch the
pair it needs when the user switches languages.

Usage (from anywhere)::

    python3 web/browser/build.py

Then deploy the printed ``web/dist`` directory, or serve it locally with::

    python3 -m http.server --directory web/dist 8000
"""

import shutil
import sys
from pathlib import Path

# Repository layout resolved relative to this file so the script works from any
# working directory.
BROWSER_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = BROWSER_DIRECTORY.parent.parent
NAHUAL_DIRECTORY = PROJECT_ROOT / "nahual"
MODELS_DIRECTORY = PROJECT_ROOT / "models"
DIST_DIRECTORY = PROJECT_ROOT / "web" / "dist"

# The language list is shared with the desktop scripts instead of being repeated
# here. This runs on CPython (not in the browser), and nahual.sign_language
# imports nothing outside the standard library, so a bare `python3
# web/browser/build.py` on a build host works with no dependencies installed.
sys.path.insert(0, str(PROJECT_ROOT))

from nahual.sign_language import SIGN_LANGUAGES  # noqa: E402

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

# The trained classifier filenames every language directory must contain. The
# browser fetches them from dist/models/<language>/, mirroring the repository's
# own models/<language>/ layout.
CLASSIFIER_FILENAMES = [
    "gesture_classifier.pkl",
    "dynamic_gesture_classifier.pkl",
]

# The MediaPipe hand model is language-independent and stays at the models/ top
# level, both in the repository and in the bundle.
HAND_LANDMARKER_FILENAME = "hand_landmarker.task"


def build() -> None:
    """Copy all runtime files into a fresh web/dist/ bundle.

    Raises:
        FileNotFoundError: If a language directory or one of its classifiers is
            missing. Failing the build is deliberate: a bundle that silently
            ships without one language's models would only surface as a 404 in
            a visitor's browser.
    """
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

    shutil.copy2(
        MODELS_DIRECTORY / HAND_LANDMARKER_FILENAME,
        DIST_DIRECTORY / "models" / HAND_LANDMARKER_FILENAME,
    )

    for language in SIGN_LANGUAGES:
        shutil.copytree(
            MODELS_DIRECTORY / language, DIST_DIRECTORY / "models" / language
        )

    verify_models()

    total_bytes = sum(
        path.stat().st_size for path in DIST_DIRECTORY.rglob("*") if path.is_file()
    )
    print(f"Built {DIST_DIRECTORY} ({total_bytes / 1_000_000:.1f} MB).")
    print("Serve locally:  python3 -m http.server --directory web/dist 8000")
    print("Then open:      http://localhost:8000")


def verify_models() -> None:
    """Fail loudly if any language is missing a classifier in the bundle.

    shutil.copytree already raises on a missing language directory, so this
    catches the subtler case: a directory that exists but lacks one of the two
    .pkl files (for example a language trained for static poses only).

    Raises:
        FileNotFoundError: If an expected classifier is not in the bundle.
    """
    for language in SIGN_LANGUAGES:
        for filename in CLASSIFIER_FILENAMES:
            bundled_path = DIST_DIRECTORY / "models" / language / filename
            if not bundled_path.is_file():
                raise FileNotFoundError(
                    f"Missing {language} classifier: expected "
                    f"{MODELS_DIRECTORY / language / filename}. Train it with "
                    f"`uv run python train.py -{language}` (and commit it) "
                    f"before building the bundle."
                )


if __name__ == "__main__":
    build()
