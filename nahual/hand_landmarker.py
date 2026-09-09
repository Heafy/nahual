"""
nahual/hand_landmarker.py

Canonical MediaPipe HandLandmarker configuration and helpers.

This module is the single source of truth for the hand-detection settings used
across the project's desktop tools: the real-time demo (``main.py``) and the
data collector (``nahual/gesture_collector.py``). Centralizing the settings here
guarantees that samples are captured under exactly the same detection parameters
they are later recognized with — if the two drifted apart, the collected
training data would no longer match what the demo sees at inference time.

The browser demo (``web/browser/app.js``) runs MediaPipe in JavaScript and cannot
import this module; it keeps its own copy of these values and must be updated in
step with :class:`HandLandmarkerConfig` whenever they change.

This module imports ``mediapipe`` and ``cv2`` at import time, so it must only be
imported by the desktop tools. The browser front-end runs MediaPipe in
JavaScript and never imports this module, so the web side stays free of the
mediapipe/opencv dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass(frozen=True)
class HandLandmarkerConfig:
    """Detection parameters for the MediaPipe HandLandmarker.

    This is the single source of truth for the hand-detection settings shared by
    ``main.py`` and ``nahual/gesture_collector.py``. The same values are mirrored
    in ``web/browser/app.js`` for the browser demo and must be changed together,
    or the desktop and browser front-ends will detect differently.

    Attributes:
        min_hand_detection_confidence: Gates the initial palm detection.
        min_hand_presence_confidence: Gates the hand-presence score inside the
            landmark model, which is a separate stage from palm detection.

    Tuning note:
        The three confidence values gate *detection* only (whether MediaPipe
        emits a hand), not the landmark geometry the classifiers consume, so
        they can be changed and tested in ``main.py`` without retraining.
        Re-sync ``web/browser/app.js`` before shipping any change.
    """

    model_asset_path: str = "models/hand_landmarker.task"
    num_hands: int = 1
    min_hand_detection_confidence: float = 0.7
    min_hand_presence_confidence: float = 0.6
    # Tuned for motion robustness: 0.7 (baseline) -> 0.5 cut mid-record dropouts
    # ~9 -> ~3 per quick gesture (chosen). 0.3 was tested and rejected — too
    # sticky/unstable, it lost the hand more often. 0.5 is the sweet spot.
    min_tracking_confidence: float = 0.5


def build_hand_landmarker(
    config: HandLandmarkerConfig = HandLandmarkerConfig(),
) -> vision.HandLandmarker:
    """Construct a HandLandmarker configured for VIDEO mode from a config.

    VIDEO running mode expects monotonically increasing timestamps and is what
    both desktop tools use for a live webcam stream. The returned object is a
    context manager, so callers can use it with
    ``with build_hand_landmarker(...) as landmarker:``.
    """
    base_options = python.BaseOptions(model_asset_path=config.model_asset_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=config.num_hands,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=config.min_hand_detection_confidence,
        min_hand_presence_confidence=config.min_hand_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_landmarks(
    landmarker: vision.HandLandmarker,
    frame_bgr: np.ndarray,
    timestamp_ms: int,
) -> vision.HandLandmarkerResult:
    """Run the hand landmarker on one OpenCV BGR frame.

    Wraps the boilerplate shared by every desktop caller: OpenCV hands back BGR
    but MediaPipe expects RGB, so the frame is converted before detection.
    ``timestamp_ms`` must increase monotonically, as VIDEO mode requires.
    """
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, timestamp_ms)
