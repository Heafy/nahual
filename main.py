"""
main.py

Real-time LSM gesture recognition demo.

Opens a webcam window with MediaPipe hand landmarks overlaid.
If a trained model exists, the predicted gesture label is displayed
on-screen.  If no model is found, the demo runs in landmark-only mode.

Press 'd' to start recording a dynamic gesture.  Press 'd' again
(or wait 3 seconds) to classify the buffered sequence with the
dynamic model.

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
from nahual.visualization import (draw_hand_connections, draw_landmark_debug,
                                  draw_prediction_overlay)

MODEL_ASSET_PATH = "models/hand_landmarker.task"
TRAINED_MODEL_PATH = Path("models/gesture_classifier.pkl")
TRAINED_DYNAMIC_MODEL_PATH = Path("models/dynamic_gesture_classifier.pkl")

# Maximum time (seconds) for a dynamic capture before auto-classifying.
DYNAMIC_CAPTURE_TIMEOUT_SECONDS: float = 3.0

# How long (seconds) to keep displaying a dynamic prediction on screen.
DYNAMIC_PREDICTION_DISPLAY_SECONDS: float = 3.0


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

    # Dynamic capture state.
    dynamic_frame_buffer: List[LandmarkFrame] = []
    dynamic_capture_active: bool = False
    dynamic_capture_start_time: float = 0.0

    # Dynamic prediction display state.
    dynamic_prediction_label: Optional[str] = None
    dynamic_prediction_confidence: float = 0.0
    dynamic_prediction_display_time: float = 0.0

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

            if result.hand_landmarks:
                draw_hand_connections(frame, result)
                # draw_landmark_debug(frame, result)

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

                    # --- Dynamic capture: buffer frames --------------------
                    if dynamic_capture_active:
                        if len(dynamic_frame_buffer) < MAX_DYNAMIC_FRAMES:
                            dynamic_frame_buffer.append(landmark_frame)

                        # Auto-stop after timeout.
                        elapsed = current_time - dynamic_capture_start_time
                        if elapsed >= DYNAMIC_CAPTURE_TIMEOUT_SECONDS:
                            result_dynamic = classify_dynamic_buffer(
                                heuristics, trainer, dynamic_frame_buffer
                            )
                            if result_dynamic is not None:
                                dynamic_prediction_label = result_dynamic[0]
                                dynamic_prediction_confidence = result_dynamic[1]
                                dynamic_prediction_display_time = current_time
                            dynamic_capture_active = False
                            dynamic_frame_buffer.clear()

                    # --- Static prediction (when not recording dynamic) ----
                    if not dynamic_capture_active and model_available and dynamic_prediction_label is None:
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
                            draw_prediction_overlay(
                                frame,
                                prediction,
                                confidence,
                                detected_handedness,
                            )
                        except Exception:
                            pass

            # --- Dynamic capture UI overlay --------------------------------
            if dynamic_capture_active:
                elapsed = current_time - dynamic_capture_start_time
                remaining = max(0.0, DYNAMIC_CAPTURE_TIMEOUT_SECONDS - elapsed)
                recording_info = (
                    f"{remaining:.1f}s remaining  |  "
                    f"{len(dynamic_frame_buffer)} frames"
                )
                draw_prediction_overlay(
                    frame,
                    "RECORDING",
                    handedness=recording_info,
                )

            # --- Display dynamic prediction for a few seconds -------------
            if dynamic_prediction_label is not None:
                time_since_prediction = current_time - dynamic_prediction_display_time
                if time_since_prediction < DYNAMIC_PREDICTION_DISPLAY_SECONDS:
                    draw_prediction_overlay(
                        frame,
                        dynamic_prediction_label,
                        dynamic_prediction_confidence,
                        None,
                    )
                else:
                    dynamic_prediction_label = None

            cv2.imshow("Nahual", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("d") and dynamic_model_available:
                if not dynamic_capture_active:
                    # Start dynamic capture.
                    dynamic_capture_active = True
                    dynamic_frame_buffer.clear()
                    dynamic_capture_start_time = current_time
                    dynamic_prediction_label = None
                else:
                    # Stop and classify.
                    result_dynamic = classify_dynamic_buffer(
                        heuristics, trainer, dynamic_frame_buffer
                    )
                    if result_dynamic is not None:
                        dynamic_prediction_label = result_dynamic[0]
                        dynamic_prediction_confidence = result_dynamic[1]
                        dynamic_prediction_display_time = current_time
                    dynamic_capture_active = False
                    dynamic_frame_buffer.clear()

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
