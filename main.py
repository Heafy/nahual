"""
main.py

Real-time LSM gesture recognition demo.

Opens a webcam window with MediaPipe hand landmarks overlaid.
If a trained model exists, the predicted gesture label is displayed
on-screen.  If no model is found, the demo runs in landmark-only mode.

Usage::

    uv run python main.py
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from nahual.gesture_heuristics import GestureHeuristics
from nahual.gesture_trainer import GestureTrainer, TrainingConfig
from nahual.visualization import (draw_hand_connections, draw_landmark_debug,
                                  draw_prediction_overlay)

MODEL_ASSET_PATH = "models/hand_landmarker.task"
TRAINED_MODEL_PATH = Path("models/gesture_classifier.pkl")


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


def main() -> None:
    """Run the real-time LSM gesture recognition demo."""
    heuristics = GestureHeuristics()

    # Load the trained classifier if one exists; otherwise run in landmark mode.
    trainer = GestureTrainer(
        TrainingConfig(model_output_directory=TRAINED_MODEL_PATH.parent)
    )
    model_available = TRAINED_MODEL_PATH.exists()
    if model_available:
        try:
            trainer.load_model(TRAINED_MODEL_PATH)
        except NotImplementedError:
            model_available = False

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

            timestamp_ms = int((time.time() - start_time) * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                draw_hand_connections(frame, result)
                # draw_landmark_debug(frame, result)

                if model_available:
                    landmark_frame = heuristics.extract_landmark_frame(
                        result, timestamp_ms
                    )
                    if landmark_frame is not None:
                        if result.handedness and result.handedness[0][0].display_name == "Left":
                            # Mirror left-hand coordinates to match right-hand training data.
                            # Negating the X-axis reflects the hand across the sagittal plane,
                            # making it geometrically equivalent to a right hand for the model.
                            landmark_frame.coordinates[:, 0] *= -1
                        features = heuristics.extract_features_static(landmark_frame)
                        try:
                            # Concatenate all heuristic outputs into the same 81-feature
                            # vector layout used by the collector:
                            #   [0:63]  normalized_coordinates (21 landmarks × 3 axes)
                            #   [63:73] finger_angles (10 joint angles)
                            #   [73:81] inter_landmark_distances (8 pairs)
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
                            detected_handedness = (
                                result.handedness[0][0].display_name
                                if result.handedness
                                else None
                            )
                            draw_prediction_overlay(
                                frame, prediction, confidence, detected_handedness
                            )
                        except NotImplementedError:
                            pass

            cv2.imshow("Nahual", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
