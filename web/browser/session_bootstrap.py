"""
web/browser/session_bootstrap.py

Bootstrap script executed *inside* Pyodide (the browser's WebAssembly Python).

The browser-only demo has no server: MediaPipe runs in JavaScript and hands
each frame's landmarks to this module, which drives the project's real
recognition code — :class:`nahual.realtime_session.RealtimeGestureSession` —
running client-side under Pyodide. This is the same Python that ``main.py`` and
the old FastAPI server used; nothing about the feature extraction, motion state
machine, or classification is reimplemented in JavaScript.

The nahual package source is written into the Pyodide virtual filesystem by
``app.js`` before this script runs; the trained ``.pkl`` models are written
under ``/session/models/<language>/`` as each language is first selected. This
module exposes three JavaScript-callable functions:

    set_language_js(language)     -> configuration JSON string
    process_frame_js(payload_json) -> overlay JSON string
    toggle_manual_js()            -> None (mirrors the desktop 'd' key)

``set_language_js`` loads that language's classifiers and replaces the session,
which is how the LSM/ASL switch in the UI works. It must be called once before
the first frame; ``app.js`` calls it with the default language at startup.
Rebuilding the session on every switch also resets the capture state machine,
so a recording in progress cannot leak across alphabets.

Payloads cross the JS/Python boundary as JSON strings so no proxy objects need
manual lifetime management on the hot per-frame path.
"""

import json
import warnings
from pathlib import Path

import numpy as np

# The trained models were pickled with scikit-learn 1.6.1 / numpy 2.0.2, while
# Pyodide ships newer builds (sklearn 1.7.x / numpy 2.2.x). The forward
# unpickle works and predictions are verified to match the desktop build, but
# scikit-learn emits an InconsistentVersionWarning on load — silence it so the
# browser console stays clean.
warnings.filterwarnings("ignore")

from nahual.gesture_heuristics import LandmarkFrame
from nahual.gesture_trainer import GestureTrainer, TrainingConfig
from nahual.realtime_session import (MOTION_START_THRESHOLD,
                                     MOTION_STOP_THRESHOLD,
                                     RealtimeGestureSession)

# Root of the model tree in the Pyodide virtual filesystem, populated by
# app.js. Mirrors the repository layout: one subdirectory per sign language,
# each holding the two classifier pickles.
MODELS_ROOT = Path("/session/models")
STATIC_MODEL_FILENAME = "gesture_classifier.pkl"
DYNAMIC_MODEL_FILENAME = "dynamic_gesture_classifier.pkl"

# Number of MediaPipe hand landmarks expected per frame.
EXPECTED_LANDMARK_COUNT = 21

# The active recognition session, replaced by set_language_js. None until the
# front-end selects a language.
_SESSION = None


def _load_trainer(models_directory):
    """Load the static and dynamic classifiers from one language's directory.

    Mirrors the graceful loading used by the desktop demo and the old FastAPI
    server: each model is loaded inside a try/except so a missing or broken
    artifact disables that prediction path instead of aborting startup.

    Args:
        models_directory: Directory holding that language's two .pkl files,
            e.g. /session/models/lsm.

    Returns:
        A tuple of (trainer, static_model_available, dynamic_model_available).
    """
    trainer = GestureTrainer(TrainingConfig(model_output_directory=models_directory))

    static_model_path = models_directory / STATIC_MODEL_FILENAME
    static_model_available = static_model_path.exists()
    if static_model_available:
        try:
            trainer.load_model(static_model_path)
        except Exception:
            static_model_available = False

    dynamic_model_path = models_directory / DYNAMIC_MODEL_FILENAME
    dynamic_model_available = dynamic_model_path.exists()
    if dynamic_model_available:
        try:
            trainer.load_dynamic_model(dynamic_model_path)
        except Exception:
            dynamic_model_available = False

    return trainer, static_model_available, dynamic_model_available


def set_language_js(language):
    """Load one language's classifiers and start a fresh session with them.

    Called by app.js at startup with the default language and again on every
    LSM/ASL switch. The module-level session is rebound in place rather than by
    re-running this script, so the PyProxy handles the front-end holds for
    process_frame_js and toggle_manual_js stay valid across a switch.

    Args:
        language: Language code naming a subdirectory of MODELS_ROOT that
            app.js has already populated (e.g. "lsm", "asl").

    Returns:
        A JSON string with the startup configuration the front-end needs: which
        of this language's models loaded, and the motion thresholds. The
        thresholds are language-independent but are reported here so the
        front-end reads the real Python constants rather than duplicating them.
    """
    global _SESSION

    trainer, static_model_available, dynamic_model_available = _load_trainer(
        MODELS_ROOT / language
    )
    _SESSION = RealtimeGestureSession(
        trainer=trainer,
        static_model_available=static_model_available,
        dynamic_model_available=dynamic_model_available,
    )

    return json.dumps(
        {
            "static_model_available": static_model_available,
            "dynamic_model_available": dynamic_model_available,
            "motion_start_threshold": MOTION_START_THRESHOLD,
            "motion_stop_threshold": MOTION_STOP_THRESHOLD,
        }
    )


def _build_landmark_frame(landmarks, timestamp_ms):
    """Build a LandmarkFrame from a client-supplied landmark list.

    Validates that the payload is a 21x3 numeric array of metric world
    landmarks before constructing the frame. Returns None for any malformed or
    missing payload so the caller treats it as "no hand visible". Mirrors
    build_landmark_frame from the old FastAPI server.

    Args:
        landmarks: A list of 21 [x, y, z] triples (MediaPipe worldLandmarks),
            or None.
        timestamp_ms: Client-supplied frame timestamp in milliseconds.

    Returns:
        A LandmarkFrame with a (21, 3) float32 coordinates array, or None.
    """
    if landmarks is None:
        return None

    try:
        coordinates = np.asarray(landmarks, dtype=np.float32)
    except (ValueError, TypeError):
        return None

    if coordinates.shape != (EXPECTED_LANDMARK_COUNT, 3):
        return None

    return LandmarkFrame(coordinates=coordinates, timestamp_ms=int(timestamp_ms))


def process_frame_js(payload_json):
    """Advance the recognition session by one frame and return overlay JSON.

    Args:
        payload_json: A JSON string with keys ``landmarks`` (21x3 list or null),
            ``handedness`` ("Left"/"Right"/null), and ``timestamp_ms`` (int).

    Returns:
        A JSON string of the overlay dict returned by
        RealtimeGestureSession.process_frame, with the model labels cast to
        plain ``str`` for JSON safety.
    """
    payload = json.loads(payload_json)
    landmark_frame = _build_landmark_frame(
        payload.get("landmarks"), payload.get("timestamp_ms", 0)
    )
    overlay = _SESSION.process_frame(landmark_frame, payload.get("handedness"))

    if overlay["static_label"] is not None:
        overlay["static_label"] = str(overlay["static_label"])
    if overlay["dynamic_label"] is not None:
        overlay["dynamic_label"] = str(overlay["dynamic_label"])

    return json.dumps(overlay)


def toggle_manual_js():
    """Start or stop a manual dynamic recording (mirrors the desktop 'd' key)."""
    _SESSION.toggle_manual()
