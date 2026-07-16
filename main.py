"""
main.py

Real-time LSM gesture recognition demo (desktop / OpenCV driver).

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
the FastAPI web server (``web/app.py``) so both front-ends behave identically.
This file only handles the desktop concerns: camera capture, MediaPipe, and
OpenCV drawing / keyboard input.

Press 'q' to quit. Press 'd' to start/stop a manual dynamic recording.
Press 'm' to toggle a motion-debug readout used to calibrate the motion
thresholds.

Usage::

    uv run python main.py
"""

import time
from pathlib import Path

import cv2

from nahual.gesture_heuristics import GestureHeuristics
from nahual.gesture_trainer import GestureTrainer, TrainingConfig
from nahual.hand_landmarker import (HandLandmarkerConfig,
                                    build_hand_landmarker, detect_landmarks)
from nahual.realtime_session import (MOTION_START_THRESHOLD,
                                     MOTION_STOP_THRESHOLD,
                                     RealtimeGestureSession)
from nahual.visualization import draw_hand_connections, draw_prediction_overlay

MODEL_ASSET_PATH = "models/hand_landmarker.task"
TRAINED_MODEL_PATH = Path("models/gesture_classifier.pkl")
TRAINED_DYNAMIC_MODEL_PATH = Path("models/dynamic_gesture_classifier.pkl")

# EMA smoothing factor for the effective-FPS readout in the motion-debug
# overlay. Heavier smoothing than the motion signal so the displayed rate is
# steady enough to read (0 < alpha <= 1; lower = smoother).
FPS_SMOOTHING_ALPHA: float = 0.1


def draw_motion_debug(
    frame,
    raw_motion: float,
    smoothed_motion: float,
    state: str,
    buffer_length: int,
    effective_fps: float,
    camera_read_ms: float,
    landmark_detect_ms: float,
    other_ms: float,
    hand_lost_during_recording: int,
) -> None:
    """Render a small motion-debug readout in the bottom-left corner.

    Shows the raw motion value, the EMA-smoothed value, the current capture
    state, the buffered frame count, the effective processing frame rate, a
    per-stage timing breakdown, and the number of frames the hand was lost
    mid-recording. The FPS and lost-frame counters diagnose whether dynamic
    dropouts are blur-driven (many lost frames at a healthy FPS) or
    throughput-driven (dropouts coinciding with a low FPS); the per-stage
    timings then localize a low FPS to the camera read vs. MediaPipe inference.

    Args:
        frame: OpenCV BGR frame to draw on.
        raw_motion: The raw per-frame motion value.
        smoothed_motion: The EMA-smoothed motion value used for thresholds.
        state: Current state of the motion-gated capture state machine
            ("IDLE" or "RECORDING").
        buffer_length: Number of frames currently buffered for dynamic
            classification.
        effective_fps: EMA-smoothed frames-per-second actually processed by the
            loop (camera read + MediaPipe + drawing).
        camera_read_ms: EMA-smoothed milliseconds spent in capture.read() per
            frame (dominated by camera exposure/bandwidth).
        landmark_detect_ms: EMA-smoothed milliseconds spent in MediaPipe hand
            detection per frame (CPU inference cost).
        other_ms: EMA-smoothed milliseconds spent on the rest of the loop
            (feature extraction, drawing, imshow, waitKey), derived so the three
            stages sum to the full frame period.
        hand_lost_during_recording: Cumulative count of frames in which the hand
            was not detected while a dynamic recording was in progress.
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
        f"fps        : {effective_fps:.1f}",
        f"ms r/d/o   : {camera_read_ms:.0f}/{landmark_detect_ms:.0f}/{other_ms:.0f}",
        f"lost@rec   : {hand_lost_during_recording}",
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

    # --- Motion-debug instrumentation state -------------------------------
    # Effective FPS is EMA-smoothed from the per-frame wall-clock delta; the
    # per-stage timers localize a low FPS to the camera vs. MediaPipe; the
    # lost-frame counter accumulates frames where the hand vanished mid-record.
    previous_frame_time: float = start_time
    smoothed_fps: float = 0.0
    smoothed_camera_read_ms: float = 0.0
    smoothed_detect_ms: float = 0.0
    smoothed_other_ms: float = 0.0
    hand_lost_during_recording_count: int = 0

    with build_hand_landmarker(
        HandLandmarkerConfig(model_asset_path=MODEL_ASSET_PATH)
    ) as landmarker:
        while True:
            read_start = time.perf_counter()
            success, frame = capture.read()
            camera_read_ms = (time.perf_counter() - read_start) * 1000.0
            if not success:
                print("Error: Failed to read frame from camera.")
                break

            current_time = time.time()
            timestamp_ms = int((current_time - start_time) * 1000)

            detect_start = time.perf_counter()
            result = detect_landmarks(landmarker, frame, timestamp_ms)
            landmark_detect_ms = (time.perf_counter() - detect_start) * 1000.0

            landmark_frame = None
            detected_handedness = None
            if result.hand_landmarks:
                draw_hand_connections(frame, result)
                landmark_frame = heuristics.extract_landmark_frame(result, timestamp_ms)
                if result.handedness:
                    detected_handedness = result.handedness[0][0].display_name

            overlay = session.process_frame(landmark_frame, detected_handedness)

            # --- Diagnostics: effective FPS + per-stage timing ------------
            frame_delta_seconds = current_time - previous_frame_time
            if frame_delta_seconds > 0:
                instantaneous_fps = 1.0 / frame_delta_seconds
                smoothed_fps = (
                    FPS_SMOOTHING_ALPHA * instantaneous_fps
                    + (1.0 - FPS_SMOOTHING_ALPHA) * smoothed_fps
                )
                # "other" is the rest of the loop period after the camera read
                # and MediaPipe detection, so the three stages sum to the full
                # frame period (the smoothed averages localize a low FPS).
                frame_period_ms = frame_delta_seconds * 1000.0
                other_ms = max(
                    0.0, frame_period_ms - camera_read_ms - landmark_detect_ms
                )
                smoothed_camera_read_ms = (
                    FPS_SMOOTHING_ALPHA * camera_read_ms
                    + (1.0 - FPS_SMOOTHING_ALPHA) * smoothed_camera_read_ms
                )
                smoothed_detect_ms = (
                    FPS_SMOOTHING_ALPHA * landmark_detect_ms
                    + (1.0 - FPS_SMOOTHING_ALPHA) * smoothed_detect_ms
                )
                smoothed_other_ms = (
                    FPS_SMOOTHING_ALPHA * other_ms
                    + (1.0 - FPS_SMOOTHING_ALPHA) * smoothed_other_ms
                )
            previous_frame_time = current_time

            # Count frames where the hand was lost while a recording was in
            # progress. Step 1's grace period keeps capture_state == "RECORDING"
            # through a brief dropout, so these are the frames it bridges.
            if not overlay["hand_visible"] and overlay["capture_state"] == "RECORDING":
                hand_lost_during_recording_count += 1

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

            # --- Recording indicator -------------------------------------
            # Let the user know a dynamic recording is in progress, whether it
            # was started manually ('d') or automatically by motion.
            if overlay["capture_state"] == "RECORDING":
                if overlay["manual_capture"]:
                    recording_info = f"manual  |  {overlay['buffer_length']} frames"
                else:
                    recording_info = (
                        f"auto  |  "
                        f"{overlay['recording_remaining_seconds']:.1f}s remaining"
                        f"  |  {overlay['buffer_length']} frames"
                    )
                stacked_y += draw_prediction_overlay(
                    frame,
                    "RECORDING",
                    handedness=recording_info,
                    y_offset=stacked_y,
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
                    smoothed_fps,
                    smoothed_camera_read_ms,
                    smoothed_detect_ms,
                    smoothed_other_ms,
                    hand_lost_during_recording_count,
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
