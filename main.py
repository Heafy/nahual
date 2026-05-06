"""
main.py

Real-time LSM gesture recognition demo.

Opens a webcam window with MediaPipe hand landmarks overlaid.
Static and dynamic gesture predictions are produced continuously and
shown stacked on screen — static on the first line (prefix "S") and
dynamic on the second line (prefix "D"). Dynamic capture is motion-gated:
the recording starts automatically when hand motion is detected and ends
when the hand becomes still (or the buffer/timeout limit is hit).

Press 'q' to quit. Press 'm' to toggle a motion-debug readout used to
calibrate the motion thresholds.

Usage::

    uv run python main.py
"""

import time
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from nahual.gesture_heuristics import (MAX_DYNAMIC_FRAMES, GestureHeuristics,
                                       LandmarkFrame)
from nahual.gesture_trainer import GestureTrainer, TrainingConfig
from nahual.visualization import draw_hand_connections, draw_prediction_overlay

MODEL_ASSET_PATH = "models/hand_landmarker.task"
TRAINED_MODEL_PATH = Path("models/gesture_classifier.pkl")
TRAINED_DYNAMIC_MODEL_PATH = Path("models/dynamic_gesture_classifier.pkl")

# Maximum time (seconds) for a dynamic capture before auto-classifying.
DYNAMIC_CAPTURE_TIMEOUT_SECONDS: float = 3.0

# How long (seconds) to keep displaying a dynamic prediction on screen.
DYNAMIC_PREDICTION_DISPLAY_SECONDS: float = 3.0

# --- Motion-gated dynamic capture tunables --------------------------------
# Smoothed per-frame motion (mean L2 distance of normalized landmarks
# between consecutive frames) above this value transitions IDLE -> RECORDING.
MOTION_START_THRESHOLD: float = 0.015

# Smoothed motion below this value, sustained for MOTION_STOP_FRAMES frames,
# transitions RECORDING -> classify. Hysteresis (stop < start) prevents
# rapid flapping near the boundary.
MOTION_STOP_THRESHOLD: float = 0.008

# Number of consecutive low-motion frames required to end a recording.
MOTION_STOP_FRAMES: int = 5

# EMA smoothing factor for the raw motion signal (0 < alpha <= 1).
# Higher = more reactive, lower = smoother.
MOTION_EMA_ALPHA: float = 0.4

# Minimum buffered frames required to attempt dynamic classification.
# Prevents spurious classifications from very short twitches.
MIN_DYNAMIC_FRAMES: int = 8

# Minimum dynamic-model confidence required to latch and display a result.
DYNAMIC_CONFIDENCE_THRESHOLD: float = 0.65


def build_hand_landmarker() -> vision.HandLandmarker:
    """Construct and return a HandLandmarker configured for VIDEO mode.

    Returns:
        A HandLandmarker context manager configured for single-hand detection.
    """
    base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.7,
    )
    return vision.HandLandmarker.create_from_options(options)


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


def draw_motion_debug(
    frame,
    raw_motion: float,
    smoothed_motion: float,
    state: str,
    buffer_length: int,
) -> None:
    """Render a small motion-debug readout in the bottom-left corner.

    Shows the raw motion value, the EMA-smoothed value, the current capture
    state, and the buffered frame count so the motion thresholds can be
    tuned interactively.

    Args:
        frame: OpenCV BGR frame to draw on.
        raw_motion: The raw per-frame motion value.
        smoothed_motion: The EMA-smoothed motion value used for thresholds.
        state: Current state of the motion-gated capture state machine
            ("IDLE" or "RECORDING").
        buffer_length: Number of frames currently buffered for dynamic
            classification.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    padding = 6
    text_color = (0, 255, 0)
    background_color = (0, 0, 0)

    lines = [
        f"raw motion : {raw_motion:.4f}",
        f"smoothed   : {smoothed_motion:.4f}",
        f"start/stop : {MOTION_START_THRESHOLD:.4f}/{MOTION_STOP_THRESHOLD:.4f}",
        f"state      : {state}  buf={buffer_length}",
    ]

    line_heights = [
        cv2.getTextSize(line, font, font_scale, thickness)[0][1] for line in lines
    ]
    line_widths = [
        cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines
    ]

    line_spacing = 4
    block_height = sum(line_heights) + line_spacing * (len(lines) - 1) + padding * 2
    block_width = max(line_widths) + padding * 2

    frame_h = frame.shape[0]
    x0 = 0
    y0 = frame_h - block_height
    cv2.rectangle(frame, (x0, y0), (x0 + block_width, frame_h), background_color, -1)

    cursor_y = y0 + padding
    for line, line_h in zip(lines, line_heights):
        cursor_y += line_h
        cv2.putText(
            frame,
            line,
            (x0 + padding, cursor_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += line_spacing


def main() -> None:
    """Run the real-time LSM gesture recognition demo."""
    heuristics = GestureHeuristics()

    # Load the trained classifiers if they exist.
    trainer = GestureTrainer(
        TrainingConfig(model_output_directory=TRAINED_MODEL_PATH.parent)
    )
    model_available = TRAINED_MODEL_PATH.exists()
    if model_available:
        try:
            trainer.load_model(TRAINED_MODEL_PATH)
        except Exception:
            model_available = False

    dynamic_model_available = TRAINED_DYNAMIC_MODEL_PATH.exists()
    if dynamic_model_available:
        try:
            trainer.load_dynamic_model(TRAINED_DYNAMIC_MODEL_PATH)
        except Exception:
            dynamic_model_available = False

    # --- Motion-gated dynamic capture state -------------------------------
    dynamic_frame_buffer: List[LandmarkFrame] = []
    capture_state: str = "IDLE"  # "IDLE" or "RECORDING"
    capture_start_time: float = 0.0
    previous_normalized: Optional[np.ndarray] = None
    smoothed_motion: float = 0.0
    raw_motion: float = 0.0
    consecutive_still_frames: int = 0

    # Latched dynamic prediction display state.
    dynamic_prediction_label: Optional[str] = None
    dynamic_prediction_confidence: float = 0.0
    dynamic_prediction_display_time: float = 0.0

    # Motion debug overlay toggle.
    show_motion_debug: bool = False

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print(
            "Error: Could not open camera. "
            "Please check your camera connection and permissions."
        )
        raise SystemExit(1)

    start_time = time.time()

    with build_hand_landmarker() as landmarker:
        while True:
            success, frame = capture.read()
            if not success:
                print("Error: Failed to read frame from camera.")
                break

            current_time = time.time()
            timestamp_ms = int((current_time - start_time) * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Per-frame transient outputs reset on every iteration.
            static_label: Optional[str] = None
            static_confidence: float = 0.0
            detected_handedness: Optional[str] = None
            hand_visible: bool = bool(result.hand_landmarks)

            if hand_visible:
                draw_hand_connections(frame, result)

                landmark_frame = heuristics.extract_landmark_frame(result, timestamp_ms)
                if landmark_frame is not None:
                    # Mirror left-hand coordinates to match right-hand training
                    # data.  Negating the X-axis reflects the hand across the
                    # sagittal plane, making it geometrically equivalent to a
                    # right hand for the model.
                    is_left_hand = (
                        result.handedness
                        and result.handedness[0][0].display_name == "Left"
                    )
                    if is_left_hand:
                        landmark_frame.coordinates[:, 0] *= -1

                    detected_handedness = (
                        result.handedness[0][0].display_name
                        if result.handedness
                        else None
                    )

                    # --- Motion signal -----------------------------------
                    current_normalized = heuristics.normalize_coordinates(
                        landmark_frame.coordinates
                    )
                    raw_motion = compute_frame_motion(
                        current_normalized, previous_normalized
                    )
                    smoothed_motion = (
                        MOTION_EMA_ALPHA * raw_motion
                        + (1.0 - MOTION_EMA_ALPHA) * smoothed_motion
                    )
                    previous_normalized = current_normalized

                    # --- Dynamic capture state machine --------------------
                    if dynamic_model_available:
                        if capture_state == "IDLE":
                            if smoothed_motion >= MOTION_START_THRESHOLD:
                                capture_state = "RECORDING"
                                capture_start_time = current_time
                                consecutive_still_frames = 0
                                dynamic_frame_buffer.clear()
                                dynamic_frame_buffer.append(landmark_frame)
                        else:  # RECORDING
                            if len(dynamic_frame_buffer) < MAX_DYNAMIC_FRAMES:
                                dynamic_frame_buffer.append(landmark_frame)

                            if smoothed_motion < MOTION_STOP_THRESHOLD:
                                consecutive_still_frames += 1
                            else:
                                consecutive_still_frames = 0

                            elapsed = current_time - capture_start_time
                            should_classify = (
                                consecutive_still_frames >= MOTION_STOP_FRAMES
                                or len(dynamic_frame_buffer) >= MAX_DYNAMIC_FRAMES
                                or elapsed >= DYNAMIC_CAPTURE_TIMEOUT_SECONDS
                            )
                            if should_classify:
                                if len(dynamic_frame_buffer) >= MIN_DYNAMIC_FRAMES:
                                    result_dynamic = classify_dynamic_buffer(
                                        heuristics, trainer, dynamic_frame_buffer
                                    )
                                    if (
                                        result_dynamic is not None
                                        and result_dynamic[1]
                                        >= DYNAMIC_CONFIDENCE_THRESHOLD
                                    ):
                                        dynamic_prediction_label = result_dynamic[0]
                                        dynamic_prediction_confidence = result_dynamic[
                                            1
                                        ]
                                        dynamic_prediction_display_time = current_time
                                capture_state = "IDLE"
                                consecutive_still_frames = 0
                                dynamic_frame_buffer.clear()

                    # --- Static prediction (always, when hand visible) ---
                    if model_available:
                        features = heuristics.extract_features_static(landmark_frame)
                        try:
                            feature_vector = np.concatenate(
                                [
                                    features.normalized_coordinates.flatten(),
                                    features.finger_angles,
                                    features.inter_landmark_distances,
                                ]
                            )
                            prediction, confidence = trainer.predict_with_confidence(
                                feature_vector
                            )
                            static_label = prediction
                            static_confidence = confidence
                        except Exception:
                            pass
            else:
                # Hand left the frame: discard any in-progress capture and
                # reset the motion reference so we don't compute distance
                # against a stale frame when the hand returns.
                if capture_state == "RECORDING":
                    capture_state = "IDLE"
                    consecutive_still_frames = 0
                    dynamic_frame_buffer.clear()
                previous_normalized = None
                smoothed_motion = 0.0
                raw_motion = 0.0

            # --- Draw stacked overlays -----------------------------------
            stacked_y = 0
            if static_label is not None:
                stacked_y += draw_prediction_overlay(
                    frame,
                    static_label,
                    static_confidence,
                    detected_handedness,
                    y_offset=stacked_y,
                    prefix="S",
                )

            if dynamic_prediction_label is not None:
                time_since_prediction = current_time - dynamic_prediction_display_time
                if time_since_prediction < DYNAMIC_PREDICTION_DISPLAY_SECONDS:
                    stacked_y += draw_prediction_overlay(
                        frame,
                        dynamic_prediction_label,
                        dynamic_prediction_confidence,
                        None,
                        y_offset=stacked_y,
                        prefix="D",
                    )
                else:
                    dynamic_prediction_label = None

            if show_motion_debug:
                draw_motion_debug(
                    frame,
                    raw_motion,
                    smoothed_motion,
                    capture_state,
                    len(dynamic_frame_buffer),
                )

            cv2.imshow("Nahual", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("m"):
                show_motion_debug = not show_motion_debug

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
