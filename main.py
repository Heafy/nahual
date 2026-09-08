"""
main.py

Real-time gesture recognition demo (desktop / OpenCV driver).

Opens a webcam window with MediaPipe hand landmarks overlaid.
Static and dynamic gesture predictions are produced continuously and
shown stacked on screen — static on the first line (prefix "S") and
dynamic on the second line (prefix "D"). Dynamic capture works two ways,
both active at the same time:

* Automatic (motion-gated): recording starts when hand motion is detected
  and ends when the hand becomes still (or the buffer/timeout limit is hit).
* Manual: press 'd' to start a recording and 'd' again to stop and classify
  it.  Useful for slow or low-amplitude gestures that do not reliably trip
  the automatic motion detector.  Pressing 'd' during an automatic recording
  stops and classifies it immediately.

The per-frame recognition logic lives in
:class:`nahual.realtime_session.RealtimeGestureSession`, which is shared with
the browser demo (``web/browser/``, running under Pyodide) so both front-ends
behave identically.
This file only handles the desktop concerns: camera capture, MediaPipe, and
OpenCV drawing / keyboard input.

Press 'q' to quit. Press 'd' to start/stop a manual dynamic recording.
Press 'm' to toggle a motion-debug readout used to calibrate the motion
thresholds.

Usage::

    uv run python main.py
    uv run python main.py -asl
"""

import time

import cv2

from nahual.gesture_heuristics import GestureHeuristics
from nahual.gesture_trainer import GestureTrainer, TrainingConfig
from nahual.hand_landmarker import (HandLandmarkerConfig,
                                    build_hand_landmarker, detect_landmarks)
from nahual.realtime_session import (MOTION_START_THRESHOLD,
                                     MOTION_STOP_THRESHOLD,
                                     RealtimeGestureSession)
from nahual.sign_language import models_directory, parse_sign_language_argument
from nahual.visualization import draw_hand_connections, draw_prediction_columns

MODEL_ASSET_PATH = "models/hand_landmarker.task"


def draw_motion_debug(
    frame,
    raw_motion: float,
    smoothed_motion: float,
    state: str,
    buffer_length: int,
) -> None:
    """Render a small motion-debug readout in the bottom-left corner.

    Shows the raw motion value, the EMA-smoothed value, the current capture
    state, and the buffered frame count. Used to calibrate
    MOTION_START_THRESHOLD/MOTION_STOP_THRESHOLD.

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
    """Run the real-time gesture recognition demo for the selected language."""
    language = parse_sign_language_argument(
        "Run the real-time Nahual gesture recognition demo."
    )
    model_output_directory = models_directory(language)
    trained_model_path = model_output_directory / "gesture_classifier.pkl"
    trained_dynamic_model_path = (
        model_output_directory / "dynamic_gesture_classifier.pkl"
    )

    heuristics = GestureHeuristics()

    # Load the trained classifiers if they exist.
    trainer = GestureTrainer(
        TrainingConfig(model_output_directory=model_output_directory)
    )
    model_available = trained_model_path.exists()
    if model_available:
        try:
            trainer.load_model(trained_model_path)
        except Exception:
            model_available = False

    dynamic_model_available = trained_dynamic_model_path.exists()
    if dynamic_model_available:
        try:
            trainer.load_dynamic_model(trained_dynamic_model_path)
        except Exception:
            dynamic_model_available = False

    # All recognition state (motion signal, capture state machine, latched
    # dynamic prediction) lives in the shared session object.
    session = RealtimeGestureSession(
        trainer=trainer,
        static_model_available=model_available,
        dynamic_model_available=dynamic_model_available,
        heuristics=heuristics,
    )

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

    with build_hand_landmarker(
        HandLandmarkerConfig(model_asset_path=MODEL_ASSET_PATH)
    ) as landmarker:
        while True:
            success, frame = capture.read()
            if not success:
                print("Error: Failed to read frame from camera.")
                break

            current_time = time.time()
            timestamp_ms = int((current_time - start_time) * 1000)

            result = detect_landmarks(landmarker, frame, timestamp_ms)

            landmark_frame = None
            detected_handedness = None
            if result.hand_landmarks:
                draw_hand_connections(frame, result)
                landmark_frame = heuristics.extract_landmark_frame(result, timestamp_ms)
                if result.handedness:
                    detected_handedness = result.handedness[0][0].display_name

            overlay = session.process_frame(landmark_frame, detected_handedness)

            # --- Draw the prediction overlay ------------------------------
            # Fixed two-column panel (static | dynamic) plus a recording row,
            # with always-visible translucent backgrounds; only the text
            # toggles, so nothing shifts on screen as signs come and go.
            draw_prediction_columns(frame, overlay)

            if show_motion_debug:
                draw_motion_debug(
                    frame,
                    overlay["raw_motion"],
                    overlay["smoothed_motion"],
                    overlay["capture_state"],
                    overlay["buffer_length"],
                )

            cv2.imshow("Nahual", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("d"):
                session.toggle_manual()
            elif key == ord("m"):
                show_motion_debug = not show_motion_debug

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
