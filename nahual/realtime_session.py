"""
nahual/realtime_session.py

Stateful, frame-by-frame LSM gesture recognition session.

This module factors the per-frame recognition logic out of ``main.py`` so it
can be reused by any front-end (the desktop OpenCV loop *and* the FastAPI web
server in ``web/``) without duplicating the motion-gated state machine.

A :class:`RealtimeGestureSession` owns all the mutable state that previously
lived as local variables inside ``main.main()`` (the dynamic frame buffer, the
capture state machine, the smoothed motion signal, the manual-capture flag,
and the latched dynamic prediction).  Feed it one :class:`LandmarkFrame` per
video frame via :meth:`RealtimeGestureSession.process_frame` and it returns a
plain dictionary describing what should be drawn on screen.
:meth:`RealtimeGestureSession.toggle_manual` mirrors the desktop 'd' key for
user-controlled dynamic recordings.

The session is intentionally agnostic about *where* the landmarks come from:
the desktop driver extracts them from MediaPipe running locally, while the web
driver receives them over a WebSocket from MediaPipe running in the browser.
Because both paths feed the same metric ``hand_world_landmarks``, the
predictions are numerically identical.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nahual.gesture_heuristics import (MAX_DYNAMIC_FRAMES, GestureHeuristics,
                                       LandmarkFrame)
from nahual.gesture_trainer import GestureTrainer

# ---------------------------------------------------------------------------
# Tunable constants (moved verbatim from main.py so there is a single source
# of truth shared by the desktop loop and the web server).
# ---------------------------------------------------------------------------

# Maximum time (seconds) for a dynamic capture before auto-classifying.
# Lowered from 3.0 once the LSM dynamic letters were confirmed to fit within
# 2 seconds; the wall-clock twin of MAX_DYNAMIC_FRAMES (60 frames at 30 fps).
DYNAMIC_CAPTURE_TIMEOUT_SECONDS: float = 2.0

# How long (seconds) to keep displaying a dynamic prediction on screen.
DYNAMIC_PREDICTION_DISPLAY_SECONDS: float = 3.0

# Smoothed per-frame motion (mean L2 distance of normalized landmarks
# between consecutive frames) above this value transitions IDLE -> RECORDING.
MOTION_START_THRESHOLD: float = 0.015

# Smoothed motion below this value, sustained for MOTION_STOP_FRAMES frames,
# transitions RECORDING -> classify. Hysteresis (stop < start) prevents
# rapid flapping near the boundary.
MOTION_STOP_THRESHOLD: float = 0.008

# Number of consecutive low-motion frames required to end a recording.
MOTION_STOP_FRAMES: int = 5

# Consecutive missing-hand frames tolerated mid-recording before the capture is
# abandoned. Bridges brief MediaPipe dropouts during fast motion so a momentary
# detection loss does not discard an in-progress gesture. ~0.17 s at 30 fps;
# tune with the FPS readout added in Step 2.
MAX_MISSING_FRAMES: int = 5

# EMA smoothing factor for the raw motion signal (0 < alpha <= 1).
# Higher = more reactive, lower = smoother.
MOTION_EMA_ALPHA: float = 0.4

# Minimum buffered frames required to attempt dynamic classification.
# Prevents spurious classifications from very short twitches.
MIN_DYNAMIC_FRAMES: int = 8

# Minimum recording duration (seconds) before stop-detection is evaluated.
# This prevents mid-gesture pauses (e.g. direction changes in "Z" or "J")
# from ending the recording prematurely. Set to cover the longest natural
# pause that can occur inside a gesture.
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
    state machine. Returns 0.0 on the first frame (no previous reference).

    Args:
        current_normalized: Normalized landmark coordinates for the current
            frame, shape (21, 3).
        previous_normalized: Normalized landmark coordinates for the previous
            frame, or None if no previous frame is available.

    Returns:
        Mean per-landmark Euclidean distance between the two frames, in the
        same normalized units as the coordinates. 0.0 if previous is None.
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
) -> Optional[Tuple[str, float]]:
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

    # Reuse the heuristics layer for the normalize-and-stack step so the
    # MAX_DYNAMIC_FRAMES cap and normalization live in a single place.  The
    # returned ExtractedFeatures.frame_sequence is the normalized, capped
    # (N_frames, 21, 3) array consumed by the statistical feature extractor.
    features = heuristics.extract_features_dynamic(dynamic_frame_buffer)
    statistical_features = heuristics.extract_statistical_features_dynamic(
        features.frame_sequence
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
        """Initialise per-stream state with sensible idle defaults.

        Args:
            trainer: GestureTrainer used for static and dynamic inference.
            static_model_available: Whether static predictions should run.
            dynamic_model_available: Whether the dynamic pipeline should run.
            heuristics: Optional GestureHeuristics instance to share.
        """
        self.trainer = trainer
        self.heuristics = heuristics or GestureHeuristics()
        self.static_model_available = static_model_available
        self.dynamic_model_available = dynamic_model_available

        # --- Motion-gated dynamic capture state ---------------------------
        self.dynamic_frame_buffer: List[LandmarkFrame] = []
        self.capture_state: str = "IDLE"  # "IDLE" or "RECORDING"
        self.manual_capture: bool = False  # True when started via toggle_manual
        self.capture_start_time: float = 0.0
        self.previous_normalized: Optional[np.ndarray] = None
        self.smoothed_motion: float = 0.0
        self.raw_motion: float = 0.0
        self.consecutive_still_frames: int = 0
        # Counts consecutive frames with no hand while RECORDING, so a brief
        # detection dropout does not immediately abandon the capture.
        self.consecutive_missing_frames: int = 0

        # --- Latched dynamic prediction display state ---------------------
        self.dynamic_prediction_label: Optional[str] = None
        self.dynamic_prediction_confidence: float = 0.0
        self.dynamic_prediction_display_time: float = 0.0

    def process_frame(
        self,
        landmark_frame: Optional[LandmarkFrame],
        handedness: Optional[str],
    ) -> Dict[str, Any]:
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
                manual_capture: bool
                buffer_length: int
                recording_remaining_seconds: Optional[float]  # None if manual
                                                              # or not recording
                raw_motion: float
                smoothed_motion: float
        """
        current_time = time.time()

        static_label: Optional[str] = None
        static_confidence: float = 0.0
        hand_visible: bool = landmark_frame is not None

        if not hand_visible:
            if self.capture_state == "RECORDING":
                # Tolerate brief dropouts: keep the buffer, capture state, and
                # motion reference so a momentary MediaPipe detection loss does
                # not discard an in-progress gesture. Only give up once the gap
                # exceeds MAX_MISSING_FRAMES consecutive frames.
                self.consecutive_missing_frames += 1
                if self.consecutive_missing_frames > MAX_MISSING_FRAMES:
                    self._reset_capture()
            else:
                # IDLE with no hand: clear the (now stale) motion reference so
                # we don't compute distance against it when the hand returns.
                self._reset_capture()
        else:
            # A visible frame ends any dropout gap.
            self.consecutive_missing_frames = 0

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

        recording_remaining_seconds: Optional[float] = None
        if self.capture_state == "RECORDING" and not self.manual_capture:
            elapsed = current_time - self.capture_start_time
            recording_remaining_seconds = max(
                0.0, DYNAMIC_CAPTURE_TIMEOUT_SECONDS - elapsed
            )

        return {
            "hand_visible": hand_visible,
            "static_label": static_label,
            "static_confidence": static_confidence,
            "handedness": handedness,
            "dynamic_label": dynamic_label,
            "dynamic_confidence": dynamic_confidence,
            "capture_state": self.capture_state,
            "manual_capture": self.manual_capture,
            "buffer_length": len(self.dynamic_frame_buffer),
            "recording_remaining_seconds": recording_remaining_seconds,
            "raw_motion": self.raw_motion,
            "smoothed_motion": self.smoothed_motion,
        }

    def toggle_manual(self) -> None:
        """Start or stop a manual dynamic recording (the desktop 'd' key).

        On IDLE, begins a user-controlled recording that ignores the automatic
        stop-detection — it ends only on the next toggle or when the buffer
        fills up.  Otherwise, stops the in-progress recording (manual or
        automatic) and classifies whatever is buffered.  A no-op when the
        dynamic model is unavailable, matching the desktop key handler.
        """
        if not self.dynamic_model_available:
            return

        current_time = time.time()
        if self.capture_state == "IDLE":
            # Manual start: begin a user-controlled dynamic recording.
            self.capture_state = "RECORDING"
            self.manual_capture = True
            self.capture_start_time = current_time
            self.consecutive_still_frames = 0
            self.dynamic_frame_buffer.clear()
        else:
            # Manual stop: classify whatever is currently buffered.  This
            # also serves as an override for an in-progress automatic
            # (motion-gated) recording.
            self._classify_buffer_and_latch(current_time)

    def _reset_capture(self) -> None:
        """Discard any in-progress dynamic capture and motion reference.

        Called when the hand leaves the frame so that motion is not computed
        against a stale frame when the hand returns.
        """
        if self.capture_state == "RECORDING":
            self.capture_state = "IDLE"
            self.manual_capture = False
            self.consecutive_still_frames = 0
            self.dynamic_frame_buffer.clear()
        self.previous_normalized = None
        self.smoothed_motion = 0.0
        self.raw_motion = 0.0
        self.consecutive_missing_frames = 0

    def _advance_dynamic_capture(
        self,
        landmark_frame: LandmarkFrame,
        current_time: float,
    ) -> None:
        """Run one step of the motion-gated dynamic capture state machine.

        Transitions IDLE -> RECORDING when motion exceeds the start threshold
        and buffers frames while recording.  Automatic (motion-started)
        recordings classify when the hand becomes still (after a minimum
        duration) or the timeout elapses; manual recordings ignore both and
        end only via toggle_manual or the buffer cap, which applies to both
        modes.

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

        # Manual recordings ignore the automatic stop-detection entirely.
        # Motion-started recordings use still-frame and timeout detection.
        if not self.manual_capture:
            # Only count still frames after the minimum recording duration
            # has passed.  This prevents mid-gesture direction-change pauses
            # (e.g. the corners of "Z") from triggering stop-detection too
            # early.
            if elapsed >= MIN_RECORDING_DURATION_SECONDS:
                if self.smoothed_motion < MOTION_STOP_THRESHOLD:
                    self.consecutive_still_frames += 1
                else:
                    self.consecutive_still_frames = 0
            else:
                self.consecutive_still_frames = 0

        # The MAX_DYNAMIC_FRAMES cap applies to both modes so the buffer can
        # never overflow; the still-frame and timeout conditions apply only
        # to automatic capture.
        should_classify = len(self.dynamic_frame_buffer) >= MAX_DYNAMIC_FRAMES or (
            not self.manual_capture
            and (
                self.consecutive_still_frames >= MOTION_STOP_FRAMES
                or elapsed >= DYNAMIC_CAPTURE_TIMEOUT_SECONDS
            )
        )
        if should_classify:
            self._classify_buffer_and_latch(current_time)

    def _classify_buffer_and_latch(self, current_time: float) -> None:
        """Classify the buffered frames, latch a confident result, and reset.

        Runs the dynamic classifier when enough frames are buffered and
        latches the prediction for the display window if its confidence
        clears DYNAMIC_CONFIDENCE_THRESHOLD, then returns the state machine
        to IDLE.

        Args:
            current_time: Wall-clock time used to start the display window.
        """
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
        self.manual_capture = False
        self.consecutive_still_frames = 0
        self.dynamic_frame_buffer.clear()

    def _predict_static(
        self,
        landmark_frame: LandmarkFrame,
    ) -> Tuple[Optional[str], float]:
        """Run the static per-frame classifier on one landmark frame.

        Builds the flat 81-feature vector via
        GestureHeuristics.flatten_static_features and runs the static model.

        Args:
            landmark_frame: The current frame's landmarks.

        Returns:
            A tuple of (label, confidence), or (None, 0.0) if inference fails.
        """
        features = self.heuristics.extract_features_static(landmark_frame)
        try:
            feature_vector = self.heuristics.flatten_static_features(features)
            prediction, confidence = self.trainer.predict_with_confidence(
                feature_vector
            )
            return prediction, confidence
        except Exception:
            return None, 0.0
