"""
web/browser/session_bootstrap.py

Bootstrap script executed *inside* Pyodide (the browser's WebAssembly Python).

The browser-only demo has no server: MediaPipe runs in JavaScript and hands
each frame's landmarks to this module, which drives the project's real
recognition code — :class:`nahual.realtime_session.RealtimeGestureSession` —
running client-side under Pyodide. This is the same Python that ``main.py`` and
the old FastAPI server used; nothing about the feature extraction, motion state
machine, or classification is reimplemented in JavaScript.

The nahual package source and the trained ``.pkl`` models are written into the
Pyodide virtual filesystem by ``app.js`` before this script runs. This module
loads the models, creates one session, and exposes three JavaScript-callable
functions:

    process_frame_js(payload_json) -> overlay JSON string
    toggle_manual_js()            -> None (mirrors the desktop 'd' key)
    status_js()                   -> configuration JSON string

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

# Locations in the Pyodide virtual filesystem that app.js populates before
# importing this module.
MODELS_DIRECTORY = Path("/session/models")
STATIC_MODEL_PATH = MODELS_DIRECTORY / "gesture_classifier.pkl"
DYNAMIC_MODEL_PATH = MODELS_DIRECTORY / "dynamic_gesture_classifier.pkl"

# Number of MediaPipe hand landmarks expected per frame.
EXPECTED_LANDMARK_COUNT = 21


def _load_trainer():
    """Load the static and dynamic classifiers if their files exist.

    Mirrors the graceful loading used by the desktop demo and the old FastAPI
    server: each model is loaded inside a try/except so a missing or broken
    artifact disables that prediction path instead of aborting startup.

    Returns:
        A tuple of (trainer, static_model_available, dynamic_model_available).
    """
    trainer = GestureTrainer(TrainingConfig(model_output_directory=MODELS_DIRECTORY))

    static_model_available = STATIC_MODEL_PATH.exists()
    if static_model_available:
        try:
            trainer.load_model(STATIC_MODEL_PATH)
        except Exception:
            static_model_available = False

    dynamic_model_available = DYNAMIC_MODEL_PATH.exists()
    if dynamic_model_available:
        try:
            trainer.load_dynamic_model(DYNAMIC_MODEL_PATH)
        except Exception:
            dynamic_model_available = False

    return trainer, static_model_available, dynamic_model_available


_TRAINER, STATIC_MODEL_AVAILABLE, DYNAMIC_MODEL_AVAILABLE = _load_trainer()

# One session per page load, exactly like one desktop run or one old WebSocket
# connection. It carries all the mutable per-frame recognition state.
_SESSION = RealtimeGestureSession(
    trainer=_TRAINER,
    static_model_available=STATIC_MODEL_AVAILABLE,
    dynamic_model_available=DYNAMIC_MODEL_AVAILABLE,
)


def _build_landmark_frame(landmarks, timestamp_ms):
    """Build a LandmarkFrame from a client-supplied landmark list.

    ``landmarks`` is a list of 21 [x, y, z] triples (MediaPipe worldLandmarks)
    or None, and ``timestamp_ms`` is the client-supplied frame timestamp in
    milliseconds. Any malformed or missing payload yields None so the caller
    treats it as "no hand visible". Mirrors build_landmark_frame from the old
    FastAPI server.
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

    ``payload_json`` carries the keys ``landmarks`` (21x3 list or null),
    ``handedness`` ("Left"/"Right"/null) and ``timestamp_ms`` (int). The reply
    is RealtimeGestureSession.process_frame's overlay dict, with the model
    labels cast to plain ``str`` for JSON safety.
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


def status_js():
    """Return startup configuration the front-end needs, as a JSON string.

    Exposes which models are loaded and the motion thresholds, mirroring the
    old ``/api/status`` endpoint so the browser readout can show the real
    Python constants.
    """
    return json.dumps(
        {
            "static_model_available": STATIC_MODEL_AVAILABLE,
            "dynamic_model_available": DYNAMIC_MODEL_AVAILABLE,
            "motion_start_threshold": MOTION_START_THRESHOLD,
            "motion_stop_threshold": MOTION_STOP_THRESHOLD,
        }
    )
