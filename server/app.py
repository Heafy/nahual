"""
server/app.py

FastAPI thin-server for the Nahual LSM gesture demo.

The browser runs MediaPipe locally to obtain hand landmarks and streams them
(tiny float arrays) over a WebSocket.  This server reuses the project's
existing Python inference code unchanged — ``GestureHeuristics``,
``GestureTrainer`` and the motion-gated ``RealtimeGestureSession`` — to turn
those landmarks into static and dynamic gesture predictions, which it sends
back for the browser to render.

No video ever leaves the user's device: only 21 landmark coordinates per
frame are transmitted.

Run locally::

    uv run uvicorn server.app:app --reload

Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nahual.gesture_heuristics import LandmarkFrame
from nahual.gesture_trainer import GestureTrainer, TrainingConfig
from nahual.realtime_session import RealtimeGestureSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIRECTORY = Path(__file__).resolve().parent.parent
MODELS_DIRECTORY = BASE_DIRECTORY / "models"
STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"

TRAINED_MODEL_PATH = MODELS_DIRECTORY / "gesture_classifier.pkl"
TRAINED_DYNAMIC_MODEL_PATH = MODELS_DIRECTORY / "dynamic_gesture_classifier.pkl"

# Number of MediaPipe hand landmarks expected per frame.
EXPECTED_LANDMARK_COUNT = 21


# ---------------------------------------------------------------------------
# Model loading (shared by every connection)
# ---------------------------------------------------------------------------


def load_trainer() -> tuple[GestureTrainer, bool, bool]:
    """Load the static and dynamic classifiers if their files exist.

    The trained models are immutable at inference time, so a single trainer
    instance is safely shared across all WebSocket connections.  Only the
    per-connection ``RealtimeGestureSession`` carries mutable state.

    Returns:
        A tuple of (trainer, static_model_available, dynamic_model_available).
    """
    trainer = GestureTrainer(TrainingConfig(model_output_directory=MODELS_DIRECTORY))

    static_model_available = TRAINED_MODEL_PATH.exists()
    if static_model_available:
        try:
            trainer.load_model(TRAINED_MODEL_PATH)
        except Exception:
            logger.exception("Failed to load static model.")
            static_model_available = False

    dynamic_model_available = TRAINED_DYNAMIC_MODEL_PATH.exists()
    if dynamic_model_available:
        try:
            trainer.load_dynamic_model(TRAINED_DYNAMIC_MODEL_PATH)
        except Exception:
            logger.exception("Failed to load dynamic model.")
            dynamic_model_available = False

    logger.info(
        "Models loaded — static: %s, dynamic: %s",
        static_model_available,
        dynamic_model_available,
    )
    return trainer, static_model_available, dynamic_model_available


app = FastAPI(title="Nahual LSM Gesture Demo")

# Shared, read-only trainer + availability flags loaded once at import time.
TRAINER, STATIC_MODEL_AVAILABLE, DYNAMIC_MODEL_AVAILABLE = load_trainer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_landmark_frame(
    landmarks: object,
    timestamp_ms: int,
) -> Optional[LandmarkFrame]:
    """Build a LandmarkFrame from a raw client-supplied landmark list.

    Validates that the payload is a 21x3 numeric array of metric world
    landmarks before constructing the frame.  Returns None for any malformed
    or missing payload so the caller can treat it as "no hand visible".

    Args:
        landmarks: A list of 21 [x, y, z] coordinate triples from the browser
            (MediaPipe ``worldLandmarks``), or None.
        timestamp_ms: Client-supplied frame timestamp in milliseconds.

    Returns:
        A LandmarkFrame with a (21, 3) float32 coordinates array, or None if
        the payload is missing or malformed.
    """
    if not landmarks:
        return None

    try:
        coordinates = np.asarray(landmarks, dtype=np.float32)
    except (ValueError, TypeError):
        return None

    if coordinates.shape != (EXPECTED_LANDMARK_COUNT, 3):
        return None

    return LandmarkFrame(coordinates=coordinates, timestamp_ms=int(timestamp_ms))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/status")
def status() -> dict:
    """Report which models are loaded so the front-end can inform the user.

    Returns:
        A dict with boolean availability flags for each model.
    """
    return {
        "static_model_available": STATIC_MODEL_AVAILABLE,
        "dynamic_model_available": DYNAMIC_MODEL_AVAILABLE,
    }


@app.websocket("/ws")
async def gesture_socket(websocket: WebSocket) -> None:
    """Per-connection gesture recognition over a WebSocket.

    Each connection owns one RealtimeGestureSession.  The browser sends one
    JSON message per video frame:

        {"landmarks": [[x, y, z], ...21...] | null,
         "handedness": "Left" | "Right" | null,
         "timestamp_ms": <int>}

    The server advances the recognition state machine and replies with the
    overlay dict produced by ``RealtimeGestureSession.process_frame``.

    Args:
        websocket: The accepted WebSocket connection.
    """
    await websocket.accept()

    session = RealtimeGestureSession(
        trainer=TRAINER,
        static_model_available=STATIC_MODEL_AVAILABLE,
        dynamic_model_available=DYNAMIC_MODEL_AVAILABLE,
    )

    try:
        while True:
            message = await websocket.receive_json()

            landmark_frame = build_landmark_frame(
                message.get("landmarks"),
                message.get("timestamp_ms", 0),
            )
            handedness = message.get("handedness")

            overlay = session.process_frame(landmark_frame, handedness)

            # Cast model labels (numpy str) to plain str for JSON safety.
            if overlay["static_label"] is not None:
                overlay["static_label"] = str(overlay["static_label"])
            if overlay["dynamic_label"] is not None:
                overlay["dynamic_label"] = str(overlay["dynamic_label"])

            await websocket.send_json(overlay)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")


@app.get("/models/hand_landmarker.task")
def hand_landmarker_model() -> FileResponse:
    """Serve the MediaPipe hand-landmarker model for in-browser detection.

    Only this single asset is exposed; the trained ``.pkl`` classifiers are
    never served because all classification happens server-side.

    Returns:
        The hand_landmarker.task file.
    """
    return FileResponse(MODELS_DIRECTORY / "hand_landmarker.task")


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page front-end.

    Returns:
        The static index.html file.
    """
    return FileResponse(STATIC_DIRECTORY / "index.html")


# Mount the remaining static assets (JS, CSS) under /static.
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")
