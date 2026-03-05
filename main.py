import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe.tasks.python.vision import hand_landmarker as mp_hand_landmarker


def draw_landmark_debug(frame, hand_landmarker_result):
    """Put landmark coordinates text in the cv2 video."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    padding = 2
    line_height = 14
    h, w = frame.shape[:2]

    LANDMARK_NAMES = (
        "WRIST",
        "THUMB_CMC",
        "THUMB_MCP",
        "THUMB_IP",
        "THUMB_TIP",
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP",
        "MIDDLE_FINGER_PIP",
        "MIDDLE_FINGER_DIP",
        "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP",
        "RING_FINGER_PIP",
        "RING_FINGER_DIP",
        "RING_FINGER_TIP",
        "PINKY_MCP",
        "PINKY_PIP",
        "PINKY_DIP",
        "PINKY_TIP",
    )

    lines = []
    for i, hand_landmarks in enumerate(hand_landmarker_result.hand_landmarks):
        handedness = hand_landmarker_result.handedness[i][0].display_name
        lines.append(f"--- {handedness} Hand ---")
        for name, lm in zip(LANDMARK_NAMES, hand_landmarks):
            lines.append(f"{name} - ({lm.x:.4f}, {lm.y:.4f}, {lm.z:.4f})")

    y = h - padding
    for line in reversed(lines):
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
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        y = y_top - padding


def main():
    """Run the webcam hand landmarker demo."""
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
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
