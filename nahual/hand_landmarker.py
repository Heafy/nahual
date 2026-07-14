"""
nahual/hand_landmarker.py

Canonical MediaPipe HandLandmarker configuration and helpers.

This module is the single source of truth for the hand-detection settings used
across the project's desktop tools: the real-time demo (``main.py``) and the
data collector (``nahual/gesture_collector.py``). Centralizing the settings here
guarantees that samples are captured under exactly the same detection parameters
they are later recognized with — if the two drifted apart, the collected
training data would no longer match what the demo sees at inference time.

The browser demo (``web/static/app.js``) runs MediaPipe in JavaScript and cannot
import this module; it keeps its own copy of these values and must be updated in
step with :class:`HandLandmarkerConfig` whenever they change.

This module imports ``mediapipe`` and ``cv2`` at import time, so it must only be
imported by the desktop tools. The thin web server (``web/app.py``) never builds
a landmarker and must stay free of the mediapipe/opencv dependencies.
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
    in ``web/static/app.js`` for the browser demo and must be changed together.

    Attributes:
        model_asset_path: Filesystem path to the ``hand_landmarker.task`` model
            asset.
        num_hands: Maximum number of hands to detect per frame.
        min_hand_detection_confidence: Minimum confidence for the initial palm
            detection to be considered successful.
        min_hand_presence_confidence: Minimum confidence for the hand-presence
            score in the landmark model.
        min_tracking_confidence: Minimum confidence for the hand-tracking to be
            considered successful between frames.
    """

    model_asset_path: str = "models/hand_landmarker.task"
    num_hands: int = 1
    min_hand_detection_confidence: float = 0.7
    min_hand_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.7


def build_hand_landmarker(
    config: HandLandmarkerConfig = HandLandmarkerConfig(),
) -> vision.HandLandmarker:
    """Construct a HandLandmarker configured for VIDEO mode from a config.

    Builds the MediaPipe options object from the given configuration and returns
    a ready-to-use landmarker. VIDEO running mode expects monotonically
    increasing timestamps and is what both desktop tools use for a live webcam
    stream. The returned object is a context manager, so callers can use it with
    ``with build_hand_landmarker(...) as landmarker:``.

    Args:
        config: The detection parameters to apply. Defaults to
            :class:`HandLandmarkerConfig` with the project's canonical values.

    Returns:
        A HandLandmarker configured for single-stream VIDEO detection.
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

    Wraps the boilerplate shared by every desktop caller: convert the OpenCV BGR
    frame to the RGB layout MediaPipe expects, wrap it in an ``mp.Image``, and
    run VIDEO-mode detection at the given timestamp.

    Args:
        landmarker: A HandLandmarker created by :func:`build_hand_landmarker`.
        frame_bgr: The OpenCV frame in BGR channel order.
        timestamp_ms: Monotonically increasing frame timestamp in milliseconds.

    Returns:
        The HandLandmarkerResult for this frame.
    """
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, timestamp_ms)
