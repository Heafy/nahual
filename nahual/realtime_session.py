"""
nahual/realtime_session.py

Stateful, frame-by-frame LSM gesture recognition session.

This module factors the per-frame recognition logic out of ``main.py`` so it
can be reused by any front-end (the desktop OpenCV loop *and* the FastAPI web
server) without duplicating the motion-gated state machine.

A :class:`RealtimeGestureSession` owns all the mutable state that previously
lived as local variables inside ``main.main()`` (the dynamic frame buffer, the
capture state machine, the smoothed motion signal, and the latched dynamic
prediction).  Feed it one :class:`LandmarkFrame` per video frame via
:meth:`RealtimeGestureSession.process_frame` and it returns a plain dictionary
describing what should be drawn on screen.

The session is intentionally agnostic about *where* the landmarks come from:
the desktop driver extracts them from MediaPipe running locally, while the web
driver receives them over a WebSocket from MediaPipe running in the browser.
Because both paths feed the same metric ``hand_world_landmarks``, the
predictions are numerically identical.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from nahual.gesture_heuristics import (MAX_DYNAMIC_FRAMES, GestureHeuristics,
                                       LandmarkFrame)
from nahual.gesture_trainer import GestureTrainer

# ---------------------------------------------------------------------------
# Tunable constants (moved verbatim from main.py so there is a single source
# of truth shared by the desktop loop and the web server).
# ---------------------------------------------------------------------------

# Maximum time (seconds) for a dynamic capture before auto-classifying.
DYNAMIC_CAPTURE_TIMEOUT_SECONDS: float = 3.0

# How long (seconds) to keep displaying a dynamic prediction on screen.
DYNAMIC_PREDICTION_DISPLAY_SECONDS: float = 3.0

# Smoothed per-frame motion above this value transitions IDLE -> RECORDING.
MOTION_START_THRESHOLD: float = 0.015

# Smoothed motion below this value, sustained for MOTION_STOP_FRAMES frames,
# transitions RECORDING -> classify.  Hysteresis (stop < start) prevents
# rapid flapping near the boundary.
MOTION_STOP_THRESHOLD: float = 0.008

# Number of consecutive low-motion frames required to end a recording.
MOTION_STOP_FRAMES: int = 5

# EMA smoothing factor for the raw motion signal (0 < alpha <= 1).
MOTION_EMA_ALPHA: float = 0.4

# Minimum buffered frames required to attempt dynamic classification.
MIN_DYNAMIC_FRAMES: int = 8

# Minimum recording duration (seconds) before stop-detection is evaluated.
MIN_RECORDING_DURATION_SECONDS: float = 1.0

# Minimum dynamic-model confidence required to latch and display a result.
DYNAMIC_CONFIDENCE_THRESHOLD: float = 0.65


# ---------------------------------------------------------------------------
# Pure helpers (moved verbatim from main.py)
# ---------------------------------------------------------------------------


def compute_frame_motion(
    current_normalized: np.ndarray,
    previous_normalized: Optional[np.ndarray],
) -> float:
    """Compute the mean L2 distance between two normalized landmark frames.

    Used as the raw motion signal driving the motion-gated dynamic capture
    state machine.  Returns 0.0 on the first frame (no previous reference).

    Args:
        current_normalized: Normalized landmark coordinates for the current
            frame, shape (21, 3).
        previous_normalized: Normalized landmark coordinates for the previous
            frame, or None if no previous frame is available.

    Returns:
        Mean per-landmark Euclidean distance between the two frames, in the
        same normalized units as the coordinates.  0.0 if previous is None.
    """
    if previous_normalized is None:
        return 0.0
    per_landmark_distances = np.linalg.norm(
        current_normalized - previous_normalized, axis=1
    )
    return float(np.mean(per_landmark_distances))


def classify_dynamic_buffer(
    heuristics: GestureHeuristics,
    trainer: GestureTrainer,
    dynamic_frame_buffer: List[LandmarkFrame],
) -> Optional[tuple[str, float]]:
    """Extract statistical features from a frame buffer and classify.

    Normalizes each buffered frame, stacks into a sequence array, computes
    statistical features, and runs the dynamic model inference.

    Args:
        heuristics: GestureHeuristics instance for feature extraction.
        trainer: GestureTrainer with a loaded dynamic model.
        dynamic_frame_buffer: List of LandmarkFrame objects captured
            during the dynamic recording session.

    Returns:
        A tuple of (label, confidence) if classification succeeds,
        or None if the buffer is empty or inference fails.
    """
    if not dynamic_frame_buffer:
        return None

    normalized_sequence = np.stack(
        [
            heuristics.normalize_coordinates(frame.coordinates)
            for frame in dynamic_frame_buffer
        ],
        axis=0,
    )  # shape (N_frames, 21, 3)

    statistical_features = heuristics.extract_statistical_features_dynamic(
        normalized_sequence
    )
    prediction, confidence = trainer.predict_dynamic_with_confidence(
        statistical_features
    )
    return prediction, confidence


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class RealtimeGestureSession:
    """Per-stream LSM recognition state machine.

    One instance corresponds to one continuous video stream (one desktop run
    or one WebSocket connection).  It must not be shared across concurrent
    streams because it carries mutable per-frame state.

    Args:
        trainer: A GestureTrainer with the static and/or dynamic models
            already loaded.
        static_model_available: Whether the static classifier is loaded and
            should be used for per-frame prediction.
        dynamic_model_available: Whether the dynamic classifier is loaded and
            the motion-gated capture pipeline should run.
        heuristics: Optional shared GestureHeuristics instance.  A new one is
            created if not supplied (it is stateless and thread-safe).
    """

    def __init__(
        self,
        trainer: GestureTrainer,
        static_model_available: bool,
        dynamic_model_available: bool,
        heuristics: Optional[GestureHeuristics] = None,
    ) -> None:
        """Initialise per-stream state with sensible idle defaults."""
        self.trainer = trainer
        self.heuristics = heuristics or GestureHeuristics()
        self.static_model_available = static_model_available
        self.dynamic_model_available = dynamic_model_available

        # --- Motion-gated dynamic capture state ---------------------------
        self.dynamic_frame_buffer: List[LandmarkFrame] = []
        self.capture_state: str = "IDLE"  # "IDLE" or "RECORDING"
        self.capture_start_time: float = 0.0
        self.previous_normalized: Optional[np.ndarray] = None
        self.smoothed_motion: float = 0.0
        self.raw_motion: float = 0.0
        self.consecutive_still_frames: int = 0

        # --- Latched dynamic prediction display state ---------------------
        self.dynamic_prediction_label: Optional[str] = None
        self.dynamic_prediction_confidence: float = 0.0
        self.dynamic_prediction_display_time: float = 0.0

    def _reset_capture(self) -> None:
        """Discard any in-progress dynamic capture and motion reference.

        Called when the hand leaves the frame so that motion is not computed
        against a stale frame when the hand returns.
        """
        if self.capture_state == "RECORDING":
            self.capture_state = "IDLE"
            self.consecutive_still_frames = 0
            self.dynamic_frame_buffer.clear()
        self.previous_normalized = None
        self.smoothed_motion = 0.0
        self.raw_motion = 0.0

    def process_frame(
        self,
        landmark_frame: Optional[LandmarkFrame],
        handedness: Optional[str],
    ) -> dict:
        """Advance the state machine by one video frame and return overlay data.

        Mirrors the body of the original ``main.main()`` loop: it runs the
        motion signal + capture state machine, the static per-frame prediction,
        and manages the latched dynamic prediction display window.

        Args:
            landmark_frame: The hand landmarks for this frame, or None if no
                hand is visible.  Coordinates are metric world landmarks of
                shape (21, 3); they are mirrored in place for left hands.
            handedness: "Left", "Right", or None.  Left-hand coordinates are
                mirrored across the sagittal plane to match the right-hand
                training data.

        Returns:
            A dictionary describing what to draw this frame:

                hand_visible: bool
                static_label: Optional[str]
                static_confidence: float
                handedness: Optional[str]
                dynamic_label: Optional[str]   # only while within display window
                dynamic_confidence: float
                capture_state: str             # "IDLE" or "RECORDING"
                raw_motion: float
                smoothed_motion: float
                buffer_length: int
        """
        current_time = time.time()

        static_label: Optional[str] = None
        static_confidence: float = 0.0
        detected_handedness: Optional[str] = handedness
        hand_visible: bool = landmark_frame is not None

        if not hand_visible:
            self._reset_capture()
        else:
            # Mirror left-hand coordinates to match right-hand training data.
            # Negating the X-axis reflects the hand across the sagittal plane,
            # making it geometrically equivalent to a right hand for the model.
            if handedness == "Left":
                landmark_frame.coordinates[:, 0] *= -1

            # --- Motion signal --------------------------------------------
            current_normalized = self.heuristics.normalize_coordinates(
                landmark_frame.coordinates
            )
            self.raw_motion = compute_frame_motion(
                current_normalized, self.previous_normalized
            )
            self.smoothed_motion = (
                MOTION_EMA_ALPHA * self.raw_motion
                + (1.0 - MOTION_EMA_ALPHA) * self.smoothed_motion
            )
            self.previous_normalized = current_normalized

            # --- Dynamic capture state machine ----------------------------
            if self.dynamic_model_available:
                self._advance_dynamic_capture(landmark_frame, current_time)

            # --- Static prediction (always, when hand visible) ------------
            if self.static_model_available:
                static_label, static_confidence = self._predict_static(landmark_frame)

        # --- Resolve latched dynamic prediction display window ------------
        dynamic_label: Optional[str] = None
        dynamic_confidence: float = 0.0
        if self.dynamic_prediction_label is not None:
            time_since_prediction = current_time - self.dynamic_prediction_display_time
            if time_since_prediction < DYNAMIC_PREDICTION_DISPLAY_SECONDS:
                dynamic_label = self.dynamic_prediction_label
                dynamic_confidence = self.dynamic_prediction_confidence
            else:
                self.dynamic_prediction_label = None

        return {
            "hand_visible": hand_visible,
            "static_label": static_label,
            "static_confidence": static_confidence,
            "handedness": detected_handedness,
            "dynamic_label": dynamic_label,
            "dynamic_confidence": dynamic_confidence,
            "capture_state": self.capture_state,
            "raw_motion": self.raw_motion,
            "smoothed_motion": self.smoothed_motion,
            "buffer_length": len(self.dynamic_frame_buffer),
        }

    def _advance_dynamic_capture(
        self,
        landmark_frame: LandmarkFrame,
        current_time: float,
    ) -> None:
        """Run one step of the motion-gated dynamic capture state machine.

        Transitions IDLE -> RECORDING when motion exceeds the start threshold,
        buffers frames while recording, and classifies the buffer when the hand
        becomes still (after a minimum duration), the buffer fills, or the
        timeout elapses.

        Args:
            landmark_frame: The current frame's landmarks to buffer.
            current_time: Wall-clock time (seconds) for this frame.
        """
        if self.capture_state == "IDLE":
            if self.smoothed_motion >= MOTION_START_THRESHOLD:
                self.capture_state = "RECORDING"
                self.capture_start_time = current_time
                self.consecutive_still_frames = 0
                self.dynamic_frame_buffer.clear()
                self.dynamic_frame_buffer.append(landmark_frame)
            return

        # RECORDING
        if len(self.dynamic_frame_buffer) < MAX_DYNAMIC_FRAMES:
            self.dynamic_frame_buffer.append(landmark_frame)

        elapsed = current_time - self.capture_start_time

        # Only count still frames after the minimum recording duration has
        # passed.  This prevents mid-gesture direction-change pauses (e.g. the
        # corners of "Z") from triggering stop-detection too early.
        if elapsed >= MIN_RECORDING_DURATION_SECONDS:
            if self.smoothed_motion < MOTION_STOP_THRESHOLD:
                self.consecutive_still_frames += 1
            else:
                self.consecutive_still_frames = 0
        else:
            self.consecutive_still_frames = 0

        should_classify = (
            self.consecutive_still_frames >= MOTION_STOP_FRAMES
            or len(self.dynamic_frame_buffer) >= MAX_DYNAMIC_FRAMES
            or elapsed >= DYNAMIC_CAPTURE_TIMEOUT_SECONDS
        )
        if should_classify:
            if len(self.dynamic_frame_buffer) >= MIN_DYNAMIC_FRAMES:
                result_dynamic = classify_dynamic_buffer(
                    self.heuristics, self.trainer, self.dynamic_frame_buffer
                )
                if (
                    result_dynamic is not None
                    and result_dynamic[1] >= DYNAMIC_CONFIDENCE_THRESHOLD
                ):
                    self.dynamic_prediction_label = result_dynamic[0]
                    self.dynamic_prediction_confidence = result_dynamic[1]
                    self.dynamic_prediction_display_time = current_time
            self.capture_state = "IDLE"
            self.consecutive_still_frames = 0
            self.dynamic_frame_buffer.clear()

    def _predict_static(
        self,
        landmark_frame: LandmarkFrame,
    ) -> tuple[Optional[str], float]:
        """Run the static per-frame classifier on one landmark frame.

        Builds the flat 81-feature vector (normalized coordinates + finger
        angles + inter-landmark distances) and runs the static model.

        Args:
            landmark_frame: The current frame's landmarks.

        Returns:
            A tuple of (label, confidence), or (None, 0.0) if inference fails.
        """
        features = self.heuristics.extract_features_static(landmark_frame)
        try:
            feature_vector = np.concatenate(
                [
                    features.normalized_coordinates.flatten(),
                    features.finger_angles,
                    features.inter_landmark_distances,
                ]
            )
            prediction, confidence = self.trainer.predict_with_confidence(
                feature_vector
            )
            return prediction, confidence
        except Exception:
            return None, 0.0
