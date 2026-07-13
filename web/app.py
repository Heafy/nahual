"""
web/app.py

FastAPI thin-server for the Nahual LSM gesture browser demo.

The browser runs MediaPipe HandLandmarker locally and streams only the
metric ``worldLandmarks`` (21 [x, y, z] triples per frame) over a
WebSocket.  This server feeds them into the project's shared recognition
code — :class:`nahual.realtime_session.RealtimeGestureSession` — and sends
the resulting overlay back for the browser to render.  No video ever
leaves the user's device and the server needs neither mediapipe nor
opencv, which keeps it responsive and the deployment image slim.

Run locally (from the repository root)::

    uv run --with-requirements web/requirements.txt python web/app.py

Then open http://localhost:8000 in a browser.  On Render, the server
binds to 0.0.0.0 and reads the port from the PORT environment variable.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Make the repository root importable so ``nahual`` resolves no matter
# which directory the server was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nahual.gesture_heuristics import LandmarkFrame
from nahual.gesture_trainer import GestureTrainer, TrainingConfig
from nahual.realtime_session import RealtimeGestureSession

logger = logging.getLogger(__name__)

MODELS_DIRECTORY = PROJECT_ROOT / "models"
STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"

TRAINED_MODEL_PATH = MODELS_DIRECTORY / "gesture_classifier.pkl"
TRAINED_DYNAMIC_MODEL_PATH = MODELS_DIRECTORY / "dynamic_gesture_classifier.pkl"
HAND_LANDMARKER_ASSET_PATH = MODELS_DIRECTORY / "hand_landmarker.task"

# Number of MediaPipe hand landmarks expected per frame.
EXPECTED_LANDMARK_COUNT = 21

DEFAULT_PORT = 8000


def load_trainer() -> Tuple[GestureTrainer, bool, bool]:
    """Load the static and dynamic classifiers if their files exist.

    Mirrors the desktop demo's graceful loading pattern: each model path is
    checked for existence and loaded inside a try/except, so a missing or
    broken artifact disables that prediction path instead of crashing the
    server.  The trained models are immutable at inference time, so a single
    trainer instance is safely shared across all WebSocket connections; only
    the per-connection RealtimeGestureSession carries mutable state.

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


app = FastAPI(title="Nahual — LSM Gesture Recognition")
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

# Shared, read-only trainer + availability flags loaded once at import time.
TRAINER, STATIC_MODEL_AVAILABLE, DYNAMIC_MODEL_AVAILABLE = load_trainer()


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


@app.get("/")
async def home_page() -> FileResponse:
    """Serve the single-page detection UI.

    Returns:
        FileResponse with web/static/index.html.
    """
    return FileResponse(STATIC_DIRECTORY / "index.html")


@app.get("/api/status")
async def status() -> dict:
    """Report which models are loaded so the front-end can inform the user.

    Returns:
        A dict with boolean availability flags for each model.
    """
    return {
        "static_model_available": STATIC_MODEL_AVAILABLE,
        "dynamic_model_available": DYNAMIC_MODEL_AVAILABLE,
    }


@app.get("/models/hand_landmarker.task")
async def hand_landmarker_model() -> FileResponse:
    """Serve the MediaPipe hand-landmarker model for in-browser detection.

    Only this single asset is exposed; the trained ``.pkl`` classifiers are
    never served because all classification happens server-side.

    Returns:
        The hand_landmarker.task file.
    """
    return FileResponse(HAND_LANDMARKER_ASSET_PATH)


@app.websocket("/ws")
async def gesture_socket(websocket: WebSocket) -> None:
    """Per-connection gesture recognition over a WebSocket.

    Each connection owns one RealtimeGestureSession.  All messages are JSON:

        {"type": "frame",
         "landmarks": [[x, y, z], ...21...] | null,
         "handedness": "Left" | "Right" | null,
         "timestamp_ms": <int>}
            -> replies with the overlay dict from process_frame.

        {"type": "toggle_manual"}
            -> starts/stops a manual dynamic recording (no reply; the state
               shows up in the next frame overlay).

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
            message_type = message.get("type", "frame")

            if message_type == "toggle_manual":
                session.toggle_manual()
                continue

            landmark_frame = build_landmark_frame(
                message.get("landmarks"),
                message.get("timestamp_ms", 0),
            )
            overlay = session.process_frame(landmark_frame, message.get("handedness"))

            # Cast model labels (numpy str) to plain str for JSON safety.
            if overlay["static_label"] is not None:
                overlay["static_label"] = str(overlay["static_label"])
            if overlay["dynamic_label"] is not None:
                overlay["dynamic_label"] = str(overlay["dynamic_label"])

            await websocket.send_json(overlay)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    # Browsers only expose the camera API on secure origins (HTTPS or
    # localhost), so the 0.0.0.0 address uvicorn logs must not be opened
    # directly — point users at localhost explicitly.
    print(
        f"Open http://localhost:{port} — browsers block camera access on "
        "http:// addresses other than localhost."
    )
    uvicorn.run(app, host="0.0.0.0", port=port)
