"""
main.py

Real-time LSM gesture recognition demo (desktop / OpenCV driver).

Opens a webcam window with MediaPipe hand landmarks overlaid.
If a trained model exists, the predicted gesture label is displayed
on-screen.  If no model is found, the demo runs in landmark-only mode.

<<<<<<< Updated upstream
Press 'd' to start recording a dynamic gesture.  Press 'd' again
(or wait 3 seconds) to classify the buffered sequence with the
dynamic model.
=======
The per-frame recognition logic lives in
:class:`nahual.realtime_session.RealtimeGestureSession`, which is shared with
the FastAPI web server so both front-ends behave identically.

Press 'q' to quit. Press 'm' to toggle a motion-debug readout used to
calibrate the motion thresholds.
>>>>>>> Stashed changes

Usage::

    uv run python main.py
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from nahual.gesture_heuristics import GestureHeuristics
from nahual.gesture_trainer import GestureTrainer, TrainingConfig
<<<<<<< Updated upstream
from nahual.visualization import (draw_hand_connections, draw_landmark_debug,
                                  draw_prediction_overlay)
=======
from nahual.realtime_session import (MOTION_START_THRESHOLD,
                                     MOTION_STOP_THRESHOLD,
                                     RealtimeGestureSession)
from nahual.visualization import draw_hand_connections, draw_prediction_overlay
>>>>>>> Stashed changes

MODEL_ASSET_PATH = "models/hand_landmarker.task"
TRAINED_MODEL_PATH = Path("models/gesture_classifier.pkl")
TRAINED_DYNAMIC_MODEL_PATH = Path("models/dynamic_gesture_classifier.pkl")

<<<<<<< Updated upstream
# Maximum time (seconds) for a dynamic capture before auto-classifying.
DYNAMIC_CAPTURE_TIMEOUT_SECONDS: float = 3.0

# How long (seconds) to keep displaying a dynamic prediction on screen.
DYNAMIC_PREDICTION_DISPLAY_SECONDS: float = 3.0

=======
>>>>>>> Stashed changes

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


def load_trainer() -> tuple[GestureTrainer, bool, bool]:
    """Load the static and dynamic classifiers if their files exist.

    Returns:
        A tuple of (trainer, static_model_available, dynamic_model_available).
        Either model may be unavailable; the session degrades gracefully.
    """
    trainer = GestureTrainer(
        TrainingConfig(model_output_directory=TRAINED_MODEL_PATH.parent)
    )

    static_model_available = TRAINED_MODEL_PATH.exists()
    if static_model_available:
        try:
            trainer.load_model(TRAINED_MODEL_PATH)
        except Exception:
            static_model_available = False

<<<<<<< Updated upstream
=======
    dynamic_model_available = TRAINED_DYNAMIC_MODEL_PATH.exists()
    if dynamic_model_available:
        try:
            trainer.load_dynamic_model(TRAINED_DYNAMIC_MODEL_PATH)
        except Exception:
            dynamic_model_available = False

    return trainer, static_model_available, dynamic_model_available


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


>>>>>>> Stashed changes
def main() -> None:
    """Run the real-time LSM gesture recognition demo."""
    heuristics = GestureHeuristics()
    trainer, static_model_available, dynamic_model_available = load_trainer()

    session = RealtimeGestureSession(
        trainer=trainer,
        static_model_available=static_model_available,
        dynamic_model_available=dynamic_model_available,
        heuristics=heuristics,
    )
<<<<<<< Updated upstream
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
=======
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
=======
            # Extract this frame's landmarks (or None) and handedness.
            landmark_frame = heuristics.extract_landmark_frame(result, timestamp_ms)
            handedness = (
                result.handedness[0][0].display_name if result.handedness else None
            )

>>>>>>> Stashed changes
            if result.hand_landmarks:
                draw_hand_connections(frame, result)
                # draw_landmark_debug(frame, result)

<<<<<<< Updated upstream
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

=======
            # Advance the shared recognition state machine.
            overlay = session.process_frame(landmark_frame, handedness)

            # --- Draw stacked overlays -----------------------------------
            stacked_y = 0
            if overlay["static_label"] is not None:
                stacked_y += draw_prediction_overlay(
                    frame,
                    overlay["static_label"],
                    overlay["static_confidence"],
                    overlay["handedness"],
                    y_offset=stacked_y,
                    prefix="S",
                )

            if overlay["dynamic_label"] is not None:
                stacked_y += draw_prediction_overlay(
                    frame,
                    overlay["dynamic_label"],
                    overlay["dynamic_confidence"],
                    None,
                    y_offset=stacked_y,
                    prefix="D",
                )

            if show_motion_debug:
                draw_motion_debug(
                    frame,
                    overlay["raw_motion"],
                    overlay["smoothed_motion"],
                    overlay["capture_state"],
                    overlay["buffer_length"],
                )

>>>>>>> Stashed changes
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
