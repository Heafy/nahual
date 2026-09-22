"""
nahual/body/landmarker.py

Canonical MediaPipe PoseLandmarker configuration and helpers for the "body"
pipeline.

This module mirrors ``nahual/hand_landmarker.py`` for pose detection. It is a
separate module on purpose: body detection settings will be tuned (BODY_PLAN.md
Phase 3) and must never be able to alter the hand-detection settings the
"hands" models were trained under. It does not import any hands module.

The pose settings are desktop-only for now; the browser demo does not run the
body pipeline, so there is no mirrored copy in ``web/browser/app.js`` yet.

This module imports ``mediapipe`` and ``cv2`` at import time, so it must only be
imported by the desktop tools.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass(frozen=True)
class PoseLandmarkerConfig:
    """Detection parameters for the MediaPipe PoseLandmarker.

    Single source of truth for the body pipeline's pose-detection settings.

    Attributes:
        model_asset_path: Filesystem path to the pose landmarker model asset.
            The variant (lite / full / heavy) is kept in the file name so it is
            always clear which variant body data was captured with.
        num_poses: Maximum number of people to detect per frame.
        min_pose_detection_confidence: Minimum confidence for the initial person
            detection to be considered successful.
        min_pose_presence_confidence: Minimum confidence for the pose-presence
            score in the landmark model.
        min_tracking_confidence: Minimum confidence for the pose tracking to be
            considered successful between frames.

    Tuning note:
        The confidences start at the MediaPipe defaults (0.5) and are tuned in
        BODY_PLAN.md Phase 3. Once body samples are collected (Phase 5) these
        values and the model variant are locked, like HandLandmarkerConfig.
    """

    model_asset_path: str = "models/pose_landmarker_full.task"
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def build_pose_landmarker(
    config: PoseLandmarkerConfig = PoseLandmarkerConfig(),
) -> vision.PoseLandmarker:
    """Construct a PoseLandmarker configured for VIDEO mode from a config.

    VIDEO running mode expects monotonically increasing timestamps, matching
    how the hand landmarker is driven from the live webcam stream. The returned
    object is a context manager, so callers can use it with
    ``with build_pose_landmarker(...) as landmarker:``.

    Args:
        config: The detection parameters to apply. Defaults to
            :class:`PoseLandmarkerConfig` with the project's canonical values.

    Returns:
        A PoseLandmarker configured for single-stream VIDEO detection.
    """
    base_options = python.BaseOptions(model_asset_path=config.model_asset_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=config.num_poses,
        min_pose_detection_confidence=config.min_pose_detection_confidence,
        min_pose_presence_confidence=config.min_pose_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_pose_landmarks(
    landmarker: vision.PoseLandmarker,
    frame_bgr: np.ndarray,
    timestamp_ms: int,
) -> vision.PoseLandmarkerResult:
    """Run the pose landmarker on one OpenCV BGR frame.

    Converts the BGR frame to the RGB ``mp.Image`` MediaPipe expects and runs
    VIDEO-mode detection at the given timestamp.

    Args:
        landmarker: A PoseLandmarker created by :func:`build_pose_landmarker`.
        frame_bgr: The OpenCV frame in BGR channel order.
        timestamp_ms: Monotonically increasing frame timestamp in milliseconds.

    Returns:
        The PoseLandmarkerResult for this frame.
    """
    # ponytail: converts the frame separately from the hand landmarker (~1 ms
    # extra in both-mode); share one conversion once core/ exists (Phase 4).
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, timestamp_ms)
