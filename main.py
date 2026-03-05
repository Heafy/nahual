import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe.tasks.python.vision import hand_landmarker as mp_hand_landmarker

# Landmark indices (MediaPipe)
# Thumb: 0 (wrist) -> 1, 2, 3, 4 (tip)
# Index: 5 -> 6 (PIP), 7, 8 (tip)
# Middle: 9 -> 10 (PIP), 11, 12 (tip)
# Ring: 13 -> 14 (PIP), 15, 16 (tip)
# Pinky: 17 -> 18 (PIP), 19, 20 (tip)

THUMB_TIP = 4
THUMB_IP = 3
WRIST = 0
FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)

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
        print("Error: Could not open camera. Please check your camera connection and permissions.")
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
