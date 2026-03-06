import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe.tasks.python.vision import hand_landmarker as mp_hand_landmarker


def draw_landmark_debug(frame, hand_landmarker_result, hand_indices=None):
    """Put landmark coordinates text in the cv2 video.

    Args:
        frame: OpenCV BGR frame to draw on.
        hand_landmarker_result: Result from HandLandmarker.detect_for_video.
        hand_indices: Optional list of indices into hand_world_landmarks to draw;
            if None, all detected hands are drawn.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    padding = 2
    line_height = 14
    h, w = frame.shape[:2]

    # Each entry is (landmark_name, bgr_color) — colors converted from RGB to BGR.
    LANDMARK_NAMES = (
        ("WRIST", (128, 128, 128)),
        ("THUMB_CMC", (128, 128, 128)),
        ("THUMB_MCP", (182, 230, 251)),
        ("THUMB_IP", (182, 230, 251)),
        ("THUMB_TIP", (182, 230, 251)),
        ("INDEX_FINGER_MCP", (128, 128, 128)),
        ("INDEX_FINGER_PIP", (127, 68, 122)),
        ("INDEX_FINGER_DIP", (127, 68, 122)),
        ("INDEX_FINGER_TIP", (127, 68, 122)),
        ("MIDDLE_FINGER_MCP", (128, 128, 128)),
        ("MIDDLE_FINGER_PIP", (39, 206, 248)),
        ("MIDDLE_FINGER_DIP", (39, 206, 248)),
        ("MIDDLE_FINGER_TIP", (39, 206, 248)),
        ("RING_FINGER_MCP", (128, 128, 128)),
        ("RING_FINGER_PIP", (59, 250, 112)),
        ("RING_FINGER_DIP", (59, 250, 112)),
        ("RING_FINGER_TIP", (59, 250, 112)),
        ("PINKY_MCP", (128, 128, 128)),
        ("PINKY_PIP", (190, 100, 45)),
        ("PINKY_DIP", (190, 100, 45)),
        ("PINKY_TIP", (190, 100, 45)),
    )

    # Each entry is (text, bgr_color).
    lines = []
    for i, hand_landmarks in enumerate(hand_landmarker_result.hand_world_landmarks):
        handedness = hand_landmarker_result.handedness[i][0].display_name
        lines.append((f"--- {handedness} Hand ---", (0, 0, 0)))
        for (name, color), lm in zip(LANDMARK_NAMES, hand_landmarks):
            lines.append((f"{name} - ({lm.x:.4f}, {lm.y:.4f}, {lm.z:.4f})", color))

    y = h - padding
    for line, color in reversed(lines):
        (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        y_top = y - text_h - padding
        cv2.rectangle(
            frame,
            (0, y_top - padding),
            (text_w + padding * 2, y + padding),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            frame,
            line,
            (padding, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y = y_top - padding


def main():
    """Run the webcam hand landmarker demo."""
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(
            "Error: Could not open camera. Please check your camera connection and permissions."
        )
        cv2.destroyAllWindows()
        raise SystemExit(1)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()

        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read frame from camera")
                break
            frame_timestamp_ms = int((time.time() - start_time) * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            hand_landmarker_result = landmarker.detect_for_video(
                mp_image, frame_timestamp_ms
            )

            # Draw hand landmarks and connections using MediaPipe drawing utils.
            if hand_landmarker_result.hand_landmarks:
                draw_landmark_debug(frame, hand_landmarker_result)
                for hand_landmarks in hand_landmarker_result.hand_landmarks:
                    drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

            cv2.imshow("Nahual", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
