"""
nahual/visualization.py

OpenCV drawing helpers for hand landmarks and gesture overlays.
These functions are shared between main.py and gesture_collector.py
to avoid code duplication.
"""

from typing import Optional

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


def draw_prediction_overlay(
    frame,
    label: str,
    confidence=None,
    handedness: Optional[str] = None,
    y_offset: int = 0,
) -> int:
    """Draw the predicted gesture label as a full-width bar on the frame.

    Renders a dark background bar with up to three stacked lines: the
    predicted label (prominent, white), an optional confidence percentage,
    and an optional detected handedness ("Left" / "Right").  The bar height
    is returned so callers can stack further bars beneath it.

    Args:
        frame: OpenCV BGR frame to draw on.
        label: Predicted gesture label string (e.g., "A", "B").
        confidence: Optional float in [0, 1] for the model prediction
            confidence, displayed as a percentage on a second line.
        handedness: Optional string indicating which hand was detected
            (e.g., "Left" or "Right"), displayed on a third line.
        y_offset: Vertical pixel offset from the top of the frame at
            which the bar should be drawn.  Defaults to 0 (top of frame).

    Returns:
        The pixel height of the drawn bar so the next bar can use it as
        its own y_offset.
    """
    # Label line style — larger, more prominent.
    label_font = cv2.FONT_HERSHEY_DUPLEX
    label_font_scale = 1.4
    label_thickness = 2
    padding = 12
    line_gap = padding  # vertical space between label and secondary line

    # Secondary line style — handedness + confidence on one line.
    secondary_font = cv2.FONT_HERSHEY_SIMPLEX
    secondary_font_scale = 0.8
    secondary_thickness = 1

    background_color = (30, 30, 30)
    label_text_color = (255, 255, 255)
    secondary_text_color = (220, 220, 220)
    low_confidence_threshold = 0.65
    low_confidence_color = (80, 80, 255)  # Bright red in BGR

    (label_w, label_h), _ = cv2.getTextSize(
        label, label_font, label_font_scale, label_thickness
    )

    # Build the single secondary line combining handedness and confidence.
    if confidence is not None and handedness is not None:
        confidence_text = f"Hand: {handedness} | Confidence: {confidence * 100:.0f}%"
    elif confidence is not None:
        confidence_text = f"{confidence * 100:.0f}%"
    elif handedness is not None:
        confidence_text = f"Hand: {handedness}"
    else:
        confidence_text = None

    confidence_h = 0
    if confidence_text is not None:
        (confidence_w, confidence_h), _ = cv2.getTextSize(
            confidence_text, secondary_font, secondary_font_scale, secondary_thickness
        )

    # Determine whether a "Low confidence" warning line must be rendered.
    low_confidence_text = None
    low_confidence_h = 0
    if confidence is not None and confidence < low_confidence_threshold:
        low_confidence_text = "Low confidence"
        (low_confidence_w, low_confidence_h), _ = cv2.getTextSize(
            low_confidence_text, secondary_font, secondary_font_scale, secondary_thickness
        )

    # Bar height grows to accommodate the optional secondary line and the
    # optional low-confidence warning line.
    bar_height = padding + label_h
    if confidence_text is not None:
        bar_height += line_gap + confidence_h
    if low_confidence_text is not None:
        bar_height += line_gap + low_confidence_h
    bar_height += padding

    cv2.rectangle(
        frame,
        (0, y_offset),
        (frame.shape[1], bar_height + y_offset),
        background_color,
        -1,
    )

    # Draw the main prediction label on the first line.
    cv2.putText(
        frame,
        label,
        (padding, y_offset + padding + label_h),
        label_font,
        label_font_scale,
        label_text_color,
        label_thickness,
        cv2.LINE_AA,
    )

    # Draw the combined handedness / confidence on the second line if available.
    if confidence_text is not None:
        cv2.putText(
            frame,
            confidence_text,
            (padding, y_offset + padding + label_h + line_gap + confidence_h),
            secondary_font,
            secondary_font_scale,
            secondary_text_color,
            secondary_thickness,
            cv2.LINE_AA,
        )

    # Draw the low-confidence warning beneath all other lines when the
    # prediction confidence falls below the defined threshold.
    if low_confidence_text is not None:
        cv2.putText(
            frame,
            low_confidence_text,
            (padding, y_offset + bar_height - padding),
            secondary_font,
            secondary_font_scale,
            low_confidence_color,
            secondary_thickness,
            cv2.LINE_AA,
        )

    return bar_height


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
    # Status text — label tier, matches draw_prediction_overlay label line.
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_font_scale = 0.8
    label_thickness = 1
    padding = 8
    label_text_color = (220, 220, 220)

    # Message badge — secondary tier, matches draw_prediction_overlay secondary line.
    secondary_font = cv2.FONT_HERSHEY_DUPLEX
    secondary_font_scale = 1.4
    secondary_thickness = 1

    background_color = (30, 30, 30)
    highlight_color = (0, 0, 220)

    status_text = (
        f"Label: {label or '(none)'}  |  "
        f"Type: {gesture_type_name}  |  "
        f"Samples: {samples_captured}"
    )

    (text_w, text_h), _ = cv2.getTextSize(
        status_text, label_font, label_font_scale, label_thickness
    )
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
        label_font,
        label_font_scale,
        label_text_color,
        label_thickness,
        cv2.LINE_AA,
    )

    if message:
        (msg_w, msg_h), _ = cv2.getTextSize(
            message, secondary_font, secondary_font_scale, secondary_thickness
        )
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
            secondary_font,
            secondary_font_scale,
            (255, 255, 255),
            secondary_thickness,
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
