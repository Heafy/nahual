"""
nahual/visualization.py

OpenCV drawing helpers for hand landmarks and gesture overlays.
These functions are shared between main.py and gesture_collector.py
to avoid code duplication.
"""

from typing import Optional

import cv2
import numpy as np
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
    prefix: Optional[str] = None,
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
        prefix: Optional short prefix prepended to the main label line
            (e.g., "S" or "D" to mark static vs. dynamic predictions).

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
    low_confidence_color = (40, 80, 255)  # Bright red in BGR

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
            low_confidence_text,
            secondary_font,
            secondary_font_scale,
            secondary_thickness,
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
    display_label = "Letter: " + label.removeprefix("letra_")
    if prefix:
        display_label = f"{prefix} {display_label}"
    cv2.putText(
        frame,
        display_label,
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


def _fit_font_scale(text, font, base_scale, thickness, max_width):
    """Return the largest scale <= base_scale whose text width fits max_width.

    OpenCV does not wrap or clip putText, so a long line would overflow its
    column. This shrinks the scale in small steps until the rendered width fits,
    letting the columns stay readable at any camera resolution or string length.

    Args:
        text: The string to be drawn.
        font: An OpenCV HERSHEY font constant.
        base_scale: The preferred (largest) font scale to start from.
        thickness: Stroke thickness used for measurement.
        max_width: Maximum allowed text width in pixels.

    Returns:
        A font scale (float) at which ``text`` fits within ``max_width``, never
        smaller than 0.1.
    """
    scale = base_scale
    while scale > 0.1:
        (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if text_width <= max_width:
            return scale
        scale -= 0.05
    return scale


def _draw_translucent_rect(frame, top_left, bottom_right, color, alpha):
    """Blend a filled rectangle into the frame so the background is lightly opaque.

    Unlike a solid ``cv2.rectangle`` fill, this alpha-blends the color into the
    region so the video shows through slightly, matching the web overlay's
    translucent bars. Coordinates are clamped to the frame bounds.

    Args:
        frame: OpenCV BGR frame to draw on (modified in place).
        top_left: (x, y) of the rectangle's top-left corner.
        bottom_right: (x, y) of the rectangle's bottom-right corner.
        color: BGR fill color tuple.
        alpha: Opacity of the fill in [0, 1]; higher is more opaque.
    """
    x0, y0 = top_left
    x1, y1 = bottom_right
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(frame.shape[1], x1)
    y1 = min(frame.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return

    region = frame[y0:y1, x0:x1]
    fill = np.zeros_like(region)
    fill[:] = color
    frame[y0:y1, x0:x1] = cv2.addWeighted(fill, alpha, region, 1.0 - alpha, 0)


def draw_prediction_columns(frame, overlay, low_confidence_threshold: float = 0.65):
    """Draw the fixed two-column prediction overlay on the desktop frame.

    Renders a stable panel that never shifts as signs come and go: two
    equal-height columns (static on the left, dynamic on the right) and a
    recording row beneath them. All three backgrounds are drawn every frame with
    a lightly-opaque black fill, and only the *text* inside appears or clears —
    the same design as the browser demo (``web/browser/``).

    The ``Static:`` / ``Dynamic:`` headers are always shown; the detected letter,
    the secondary confidence line, and the low-confidence warning appear only
    when there is a result. Each column reserves three text lines (label +
    secondary + warning slot) so the box height is fixed even when the warning
    line is absent. Long lines are auto-fit to their column width.

    Args:
        frame: OpenCV BGR frame to draw on (modified in place).
        overlay: The dict returned by
            :meth:`nahual.realtime_session.RealtimeGestureSession.process_frame`.
        low_confidence_threshold: Static confidence below which the red
            "Low confidence" warning line is shown.
    """
    frame_width = frame.shape[1]

    # --- Style (reuse the project's existing overlay colors) --------------
    label_font = cv2.FONT_HERSHEY_DUPLEX
    secondary_font = cv2.FONT_HERSHEY_SIMPLEX
    label_thickness = 2
    secondary_thickness = 1
    label_base_scale = 1.1
    secondary_base_scale = 0.55
    padding = 10
    line_gap = 8
    column_gap = 6

    background_color = (30, 30, 30)
    background_alpha = 0.85  # lightly opaque; the video shows through ~15%
    label_color = (255, 255, 255)
    secondary_color = (220, 220, 220)
    low_confidence_color = (40, 80, 255)  # bright red in BGR, matches the old bar

    # Reserve line heights from the base scales so the panel size is stable even
    # when per-line auto-fit shrinks an individual (long) line.
    label_height = cv2.getTextSize("Ag", label_font, label_base_scale, label_thickness)[
        0
    ][1]
    secondary_height = cv2.getTextSize(
        "Ag", secondary_font, secondary_base_scale, secondary_thickness
    )[0][1]

    # Columns reserve three text lines (label + secondary + warning slot); the
    # recording row reserves one.
    column_height = (
        padding
        + label_height
        + line_gap
        + secondary_height
        + line_gap
        + secondary_height
        + padding
    )
    recording_height = padding + secondary_height + padding

    left_x0 = 0
    left_x1 = frame_width // 2 - column_gap // 2
    right_x0 = frame_width // 2 + column_gap // 2
    right_x1 = frame_width

    # --- Backgrounds (always drawn, so the layout never shifts) -----------
    _draw_translucent_rect(
        frame,
        (left_x0, 0),
        (left_x1, column_height),
        background_color,
        background_alpha,
    )
    _draw_translucent_rect(
        frame,
        (right_x0, 0),
        (right_x1, column_height),
        background_color,
        background_alpha,
    )
    _draw_translucent_rect(
        frame,
        (0, column_height),
        (frame_width, column_height + recording_height),
        background_color,
        background_alpha,
    )

    def draw_column(x0, x1, header, letter, secondary_text, warning_text):
        """Draw one column's three text lines within [x0, x1]."""
        inner_width = x1 - x0 - padding * 2

        # Line 1: header (always) plus the detected letter (when present).
        line_one = header if letter is None else f"{header} {letter}"
        scale = _fit_font_scale(
            line_one, label_font, label_base_scale, label_thickness, inner_width
        )
        y = padding + label_height
        cv2.putText(
            frame,
            line_one,
            (x0 + padding, y),
            label_font,
            scale,
            label_color,
            label_thickness,
            cv2.LINE_AA,
        )

        # Line 2: secondary confidence line (only when there is a detection).
        y += line_gap + secondary_height
        if secondary_text:
            scale = _fit_font_scale(
                secondary_text,
                secondary_font,
                secondary_base_scale,
                secondary_thickness,
                inner_width,
            )
            cv2.putText(
                frame,
                secondary_text,
                (x0 + padding, y),
                secondary_font,
                scale,
                secondary_color,
                secondary_thickness,
                cv2.LINE_AA,
            )

        # Line 3: reserved warning slot; text only when a warning is present.
        y += line_gap + secondary_height
        if warning_text:
            scale = _fit_font_scale(
                warning_text,
                secondary_font,
                secondary_base_scale,
                secondary_thickness,
                inner_width,
            )
            cv2.putText(
                frame,
                warning_text,
                (x0 + padding, y),
                secondary_font,
                scale,
                low_confidence_color,
                secondary_thickness,
                cv2.LINE_AA,
            )

    # --- Static column -----------------------------------------------------
    static_letter = None
    static_secondary = None
    static_warning = None
    if overlay["static_label"] is not None:
        static_letter = str(overlay["static_label"]).removeprefix("letra_")
        confidence_percent = f"{overlay['static_confidence'] * 100:.0f}"
        handedness = overlay["handedness"]
        if handedness:
            static_secondary = f"Hand: {handedness} | Confidence: {confidence_percent}%"
        else:
            static_secondary = f"Confidence: {confidence_percent}%"
        if overlay["static_confidence"] < low_confidence_threshold:
            static_warning = "Low confidence"
    draw_column(
        left_x0, left_x1, "Static:", static_letter, static_secondary, static_warning
    )

    # --- Dynamic column ----------------------------------------------------
    dynamic_letter = None
    dynamic_secondary = None
    if overlay["dynamic_label"] is not None:
        dynamic_letter = str(overlay["dynamic_label"]).removeprefix("letra_")
        confidence_percent = f"{overlay['dynamic_confidence'] * 100:.0f}"
        dynamic_secondary = (
            f"Confidence: {confidence_percent}%  |  "
            f"{overlay['dynamic_frame_count']} frames"
        )
    draw_column(right_x0, right_x1, "Dynamic:", dynamic_letter, dynamic_secondary, None)

    # --- Recording row (text only while a recording is in progress) -------
    if overlay["capture_state"] == "RECORDING":
        if overlay["manual_capture"]:
            recording_text = f"manual  |  {overlay['buffer_length']} frames"
        else:
            recording_text = (
                f"auto  |  {overlay['recording_remaining_seconds']:.1f}s remaining"
                f"  |  {overlay['buffer_length']} frames"
            )
        inner_width = frame_width - padding * 2
        scale = _fit_font_scale(
            recording_text,
            secondary_font,
            secondary_base_scale,
            secondary_thickness,
            inner_width,
        )
        y = column_height + padding + secondary_height
        cv2.putText(
            frame,
            recording_text,
            (padding, y),
            secondary_font,
            scale,
            secondary_color,
            secondary_thickness,
            cv2.LINE_AA,
        )
