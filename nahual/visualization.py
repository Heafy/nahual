"""
nahual/visualization.py

OpenCV drawing helpers for hand landmarks and gesture overlays.
These functions are shared between main.py and gesture_collector.py
to avoid code duplication.
"""

import cv2
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe.tasks.python.vision import hand_landmarker as mp_hand_landmarker

# Each entry is (landmark_name, bgr_color).
# Colors are BGR-converted versions of finger-part color conventions.
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


def draw_landmark_debug(frame, hand_landmarker_result, hand_indices=None):
    """Put landmark coordinate text on the OpenCV frame.

    Renders each landmark's world-coordinate values as colored text lines
    at the bottom of the frame, with a white background for readability.

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
    h, w = frame.shape[:2]

    lines = []
    for i, hand_landmarks in enumerate(hand_landmarker_result.hand_world_landmarks):
        if hand_indices is not None and i not in hand_indices:
            continue
        handedness = hand_landmarker_result.handedness[i][0].display_name
        lines.append((f"--- {handedness} Hand ---", (0, 0, 0)))
        for (name, color), landmark in zip(LANDMARK_NAMES, hand_landmarks):
            lines.append(
                (
                    f"{name} - ({landmark.x:.4f}, {landmark.y:.4f}, {landmark.z:.4f})",
                    color,
                )
            )

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


def draw_hand_connections(frame, hand_landmarker_result):
    """Draw MediaPipe hand skeleton connections on the frame.

    Uses MediaPipe's built-in drawing utilities to render the hand graph
    (bones between landmarks) with the default color style.

    Args:
        frame: OpenCV BGR frame to draw on.
        hand_landmarker_result: Result from HandLandmarker.detect_for_video.
    """
    for hand_landmarks in hand_landmarker_result.hand_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )


def draw_prediction_overlay(frame, label, confidence=None):
    """Draw the predicted gesture label prominently on the frame.

    Renders a large text banner at the top of the frame showing the
    predicted LSM letter on the main line, with the model confidence
    percentage displayed as a smaller sub-line below it.

    Args:
        frame: OpenCV BGR frame to draw on.
        label: Predicted gesture label string (e.g., "A", "B").
        confidence: Optional float in [0, 1] for the model prediction confidence,
            displayed as a percentage on a sub-line below the label.
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2.0
    thickness = 3
    padding = 10

    sub_font = cv2.FONT_HERSHEY_SIMPLEX
    sub_font_scale = 0.6
    sub_thickness = 1
    sub_gap = 6  # vertical gap between the two text lines

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Measure the optional confidence sub-line.
    sub_text = f"{confidence * 100:.0f}% confidence" if confidence is not None else None
    sub_h = 0
    sub_w = 0
    if sub_text is not None:
        (sub_w, sub_h), _ = cv2.getTextSize(
            sub_text, sub_font, sub_font_scale, sub_thickness
        )

    # Background rectangle covers both lines.
    box_width = max(text_w, sub_w) + padding * 2
    box_height = text_h + baseline + padding * 2 + (sub_gap + sub_h if sub_text else 0)
    cv2.rectangle(frame, (0, 0), (box_width, box_height), (0, 0, 0), -1)

    # Draw the main prediction label.
    cv2.putText(
        frame,
        label,
        (padding, text_h + padding),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )

    # Draw the confidence sub-line if available.
    if sub_text is not None:
        sub_y = text_h + padding + baseline + sub_gap + sub_h
        cv2.putText(
            frame,
            sub_text,
            (padding, sub_y),
            sub_font,
            sub_font_scale,
            (180, 180, 180),
            sub_thickness,
            cv2.LINE_AA,
        )


def draw_status_bar(
    frame,
    label,
    gesture_type_name,
    samples_captured,
    message=None,
    y_offset: int = 0,
) -> int:
    """Draw a status bar on the collector frame at a given vertical offset.

    Shows the current label, gesture type, sample count, and an optional
    message (e.g., countdown, recording indicator).  The bar height is
    returned so callers can stack multiple bars without hard-coding pixel
    positions.

    Args:
        frame: OpenCV BGR frame to draw on.
        label: Current label string, or "(none)" if not set.
        gesture_type_name: String name of the gesture type ("STATIC" or "DYNAMIC").
        samples_captured: Integer count of samples captured this session.
        message: Optional string shown in a highlighted box (e.g., "RECORDING").
        y_offset: Vertical pixel offset from the top of the frame at which the
            bar should be drawn.  Defaults to 0 (top of frame).

    Returns:
        The pixel height of the drawn bar so the next bar can use it as its
        own y_offset.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    padding = 6
    background_color = (30, 30, 30)
    text_color = (220, 220, 220)
    highlight_color = (0, 0, 220)

    status_text = (
        f"Label: {label or '(none)'}  |  "
        f"Type: {gesture_type_name}  |  "
        f"Samples: {samples_captured}"
    )

    (text_w, text_h), _ = cv2.getTextSize(status_text, font, font_scale, thickness)
    bar_height = text_h + padding * 2

    cv2.rectangle(
        frame,
        (0, y_offset),
        (frame.shape[1], bar_height + y_offset),
        background_color,
        -1,
    )
    cv2.putText(
        frame,
        status_text,
        (padding, text_h + padding + y_offset),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

    if message:
        (msg_w, msg_h), _ = cv2.getTextSize(message, font, font_scale, thickness)
        msg_x = frame.shape[1] - msg_w - padding * 2
        cv2.rectangle(
            frame,
            (msg_x - padding, y_offset),
            (frame.shape[1], bar_height + y_offset),
            highlight_color,
            -1,
        )
        cv2.putText(
            frame,
            message,
            (msg_x, text_h + padding + y_offset),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return bar_height


def draw_hint_bar(frame, hint_text: str, y_offset: int = 0) -> int:
    """Draw a keyboard-hint and hand-detection status bar on the frame.

    Renders a full-width background bar with the hint text using the same
    visual style as draw_status_bar (font, colors, padding).  The bar is
    positioned at y_offset from the top, so multiple bars can be stacked by
    passing the return value of a previous bar call as the next y_offset.

    Args:
        frame: OpenCV BGR frame to draw on.
        hint_text: Full hint string to display (e.g., "HAND: 95%  |  [l] label ...").
        y_offset: Vertical pixel offset from the top of the frame at which the
            bar should be drawn.  Defaults to 0 (top of frame).

    Returns:
        The pixel height of the drawn bar so the next bar can use it as its
        own y_offset.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    padding = 6
    background_color = (30, 30, 30)
    text_color = (220, 220, 220)

    (text_w, text_h), _ = cv2.getTextSize(hint_text, font, font_scale, thickness)
    bar_height = text_h + padding * 2

    cv2.rectangle(
        frame,
        (0, y_offset),
        (frame.shape[1], bar_height + y_offset),
        background_color,
        -1,
    )
    cv2.putText(
        frame,
        hint_text,
        (padding, text_h + padding + y_offset),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

    return bar_height
