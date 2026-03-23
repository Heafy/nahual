"""
nahual/gesture_collector.py

Interactive data collection tool for LSM gesture samples.

Opens a webcam window with live MediaPipe hand landmarks overlaid.
The user types a label in the terminal, then captures static or dynamic
gesture samples via keyboard.  Collected samples are saved as .npy files
in a structured directory tree under a configurable data root.

Directory layout produced::

    data/
      static/<label>/<uuid>.npy     # shape (81,), dtype float32
                                    #   [0:63]  normalized_coordinates (21×3 flattened)
                                    #   [63:73] finger_angles (10 joint angles)
                                    #   [73:81] inter_landmark_distances (8 pairs)
      dynamic/<label>/<uuid>.npy    # shape (N_frames, 21, 3), dtype float32

Each label directory also contains a manifest.json that records capture
timestamps and session IDs for traceability.

Keyboard controls (also shown as an overlay on the video frame):
    l   -- Enter a new label (pauses video; uses terminal input)
    s   -- Capture one STATIC sample from the current frame
    d   -- Start / stop DYNAMIC capture (auto-stops after the configured duration)
    q   -- Quit the collector
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from nahual.gesture_heuristics import (MAX_DYNAMIC_FRAMES, GestureHeuristics,
                                       GestureType, LandmarkFrame)
from nahual.visualization import (draw_hand_connections, draw_landmark_debug,
                                  draw_status_bar)

# ---------------------------------------------------------------------------
# Configuration and session dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CollectorConfig:
    """All tunable parameters for a data collection session.

    Attributes:
        model_asset_path: Path to the hand_landmarker.task file.
        data_root_directory: Root directory under which static/ and dynamic/
            subdirectories are created.
        camera_device_index: OpenCV camera index (0 = default webcam).
        target_fps: Desired capture framerate; controls loop sleep timing.
        dynamic_capture_duration_seconds: Maximum recording time for dynamic
            gesture capture.  Hard-capped by MAX_DYNAMIC_FRAMES as well.
        countdown_seconds: On-screen countdown before dynamic capture starts.
        window_name: Title of the OpenCV display window.
        show_landmark_debug: Whether to render coordinate text overlay.
    """

    model_asset_path: str = "models/hand_landmarker.task"
    data_root_directory: Path = Path("data")
    camera_device_index: int = 0
    target_fps: int = 30
    dynamic_capture_duration_seconds: float = 3.0
    countdown_seconds: int = 3
    window_name: str = "Nahual - Collector"
    show_landmark_debug: bool = False


@dataclass
class CollectionSession:
    """Metadata for the ongoing data-collection run.

    Attributes:
        session_id: UUID4 string, unique per run.
        label: The active gesture label (e.g., "A", "espacio").
        samples_captured: Running count of saved samples.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: Optional[str] = None
    samples_captured: int = 0


# ---------------------------------------------------------------------------
# GestureCollector
# ---------------------------------------------------------------------------


class GestureCollector:
    """Interactive tool for collecting labeled LSM gesture samples.

    Runs a live webcam feed with MediaPipe hand landmark overlay.
    The operator enters a label via the terminal, then uses keyboard
    shortcuts to trigger static or dynamic captures.  All samples are
    saved to disk as numpy .npy files.

    Args:
        config: CollectorConfig instance.  Defaults are applied if None.
    """

    def __init__(self, config: Optional[CollectorConfig] = None) -> None:
        """Initialize the collector with optional configuration.

        Args:
            config: CollectorConfig dataclass.  Defaults are used if None.
        """
        self.config = config or CollectorConfig()
        self.heuristics = GestureHeuristics()
        self.session = CollectionSession()

        self._is_capturing_dynamic: bool = False
        self._dynamic_start_time: float = 0.0
        self._dynamic_frame_buffer: List[LandmarkFrame] = []
        self._current_gesture_type: GestureType = GestureType.STATIC

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the webcam window and enter the main collection loop.

        Blocks until the user presses 'q'.  All captures are written to
        disk before this method returns.

        Raises:
            SystemExit: If the camera or the landmarker model cannot be opened.
        """
        base_options = python.BaseOptions(model_asset_path=self.config.model_asset_path)
        landmarker_options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.7,
        )

        capture = cv2.VideoCapture(self.config.camera_device_index)
        if not capture.isOpened():
            print("Error: Could not open camera.")
            raise SystemExit(1)

        self._print_instructions()
        start_time = time.time()

        with vision.HandLandmarker.create_from_options(
            landmarker_options
        ) as landmarker:
            while True:
                success, frame = capture.read()
                if not success:
                    print("Error: Failed to read frame from camera.")
                    break

                timestamp_ms = int((time.time() - start_time) * 1000)
                result = self._detect_landmarks(landmarker, frame, timestamp_ms)

                current_landmark_frame: Optional[LandmarkFrame] = None
                if result and result.hand_landmarks:
                    current_landmark_frame = self.heuristics.extract_landmark_frame(
                        result, timestamp_ms
                    )
                    if self.config.show_landmark_debug:
                        draw_landmark_debug(frame, result)
                    draw_hand_connections(frame, result)

                # Buffer frames during dynamic capture.
                if self._is_capturing_dynamic and current_landmark_frame is not None:
                    self._buffer_dynamic_frame(current_landmark_frame)
                    elapsed = time.time() - self._dynamic_start_time
                    if elapsed >= self.config.dynamic_capture_duration_seconds:
                        self.stop_dynamic_capture_and_save()

                self._draw_overlay(frame, current_landmark_frame)
                cv2.imshow(self.config.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("l"):
                    self._prompt_label_via_terminal()
                elif key == ord("s"):
                    if current_landmark_frame is not None:
                        saved_path = self.capture_static_sample(current_landmark_frame)
                        print(f"[collector] Static sample saved: {saved_path}")
                    else:
                        print(
                            "[collector] No hand detected — static sample not captured."
                        )
                elif key == ord("d"):
                    if not self._is_capturing_dynamic:
                        self.start_dynamic_capture()
                    else:
                        self.stop_dynamic_capture_and_save()

        capture.release()
        cv2.destroyAllWindows()
        print(
            f"[collector] Session ended.  Total samples: {self.session.samples_captured}"
        )

    def set_label(self, label: str) -> None:
        """Set the active gesture label and create its output directory.

        Args:
            label: Non-empty alphanumeric label string (e.g., "A", "espacio").
        """
        label = label.strip()
        if not label:
            print("[collector] Label cannot be empty.")
            return

        self.session.label = label
        self._ensure_label_directories(label)
        print(f"[collector] Label set to '{label}'.")

    def capture_static_sample(self, landmark_frame: LandmarkFrame) -> Path:
        """Extract features from one frame and write to disk.

        Args:
            landmark_frame: The LandmarkFrame to save.

        Returns:
            Path of the written .npy file.

        Raises:
            RuntimeError: If no label has been set via set_label.
        """
        self._require_label()
        features = self.heuristics.extract_features_static(landmark_frame)

        # Concatenate all heuristic outputs into a single flat feature vector:
        #   - normalized_coordinates: wrist-centred, palm-scale-invariant positions
        #     (21 landmarks × 3 axes = 63 values)
        #   - finger_angles: per-joint flexion angles derived from FINGER_JOINT_TRIPLETS
        #     (10 values, one per joint pair)
        #   - inter_landmark_distances: Euclidean distances between DEFAULT_LANDMARK_PAIRS
        #     (8 values)
        # Total: 81 float32 features saved as a 1-D array of shape (81,).
        normalized_coordinates = features.normalized_coordinates.flatten()
        finger_angles = features.finger_angles
        inter_landmark_distances = features.inter_landmark_distances

        feature_vector = np.concatenate(
            [normalized_coordinates, finger_angles, inter_landmark_distances]
        )

        nc_end = len(normalized_coordinates)
        fa_end = nc_end + len(finger_angles)
        ild_end = fa_end + len(inter_landmark_distances)

        output_path = self._build_output_path(GestureType.STATIC)
        metadata = {
            "file": output_path.name,
            "session_id": self.session.session_id,
            "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gesture_type": GestureType.STATIC.name.lower(),
            "feature_layout": {
                "normalized_coordinates": [0, nc_end],
                "finger_angles": [nc_end, fa_end],
                "inter_landmark_distances": [fa_end, ild_end],
                "total_features": ild_end,
            },
        }
        self._write_sample(feature_vector, output_path, metadata)
        self.session.samples_captured += 1
        return output_path

    def start_dynamic_capture(self) -> None:
        """Begin buffering frames for a dynamic gesture sample.

        Clears any existing buffer and sets the recording flag so that
        subsequent frames are accumulated.
        """
        if self.session.label is None:
            print("[collector] Set a label first (press 'l').")
            return
        self._dynamic_frame_buffer = []
        self._is_capturing_dynamic = True
        self._dynamic_start_time = time.time()
        print(
            f"[collector] Dynamic capture started.  "
            f"Recording for up to {self.config.dynamic_capture_duration_seconds:.1f}s ..."
        )

    def stop_dynamic_capture_and_save(self) -> Optional[Path]:
        """Stop buffering, process accumulated frames, and save to disk.

        Applies GestureHeuristics.extract_features_dynamic to the buffer,
        then saves the resulting frame_sequence array.

        Returns:
            Path of the written .npy file, or None if the buffer was empty.

        Raises:
            RuntimeError: If no label has been set.
        """
        self._is_capturing_dynamic = False
        if not self._dynamic_frame_buffer:
            print("[collector] Dynamic buffer is empty — no sample saved.")
            return None

        self._require_label()
        features = self.heuristics.extract_features_dynamic(self._dynamic_frame_buffer)
        output_path = self._build_output_path(GestureType.DYNAMIC)
        metadata = {
            "file": output_path.name,
            "session_id": self.session.session_id,
            "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gesture_type": GestureType.DYNAMIC.name.lower(),
            "frame_count": len(self._dynamic_frame_buffer),
        }
        self._write_sample(features.frame_sequence, output_path, metadata)
        self._dynamic_frame_buffer = []
        self.session.samples_captured += 1
        print(f"[collector] Dynamic sample saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_landmarks(self, landmarker, frame: np.ndarray, timestamp_ms: int):
        """Convert a BGR frame to RGB and run the hand landmarker.

        Args:
            landmarker: MediaPipe HandLandmarker context object.
            frame: OpenCV BGR frame.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            HandLandmarkerResult, or None if detection fails.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return landmarker.detect_for_video(mp_image, timestamp_ms)

    def _buffer_dynamic_frame(self, landmark_frame: LandmarkFrame) -> None:
        """Append a frame to the dynamic capture buffer.

        Silently discards frames beyond MAX_DYNAMIC_FRAMES to honour
        the hard cap.

        Args:
            landmark_frame: LandmarkFrame to append.
        """
        if len(self._dynamic_frame_buffer) < MAX_DYNAMIC_FRAMES:
            self._dynamic_frame_buffer.append(landmark_frame)

    def _build_output_path(self, gesture_type: GestureType) -> Path:
        """Construct a unique output path for a new sample file.

        Args:
            gesture_type: STATIC or DYNAMIC, determines the subdirectory.

        Returns:
            Path: data/<type>/<label>/<uuid4>.npy
        """
        type_directory = "static" if gesture_type == GestureType.STATIC else "dynamic"
        label_directory = (
            self.config.data_root_directory / type_directory / self.session.label
        )
        label_directory.mkdir(parents=True, exist_ok=True)
        return label_directory / f"{uuid.uuid4()}.npy"

    def _write_sample(
        self,
        array: np.ndarray,
        output_path: Path,
        metadata: Dict,
    ) -> None:
        """Atomically write a numpy array and append to the manifest.

        Uses a temporary file + os.replace to avoid partial writes if the
        process is interrupted mid-write.

        Args:
            array: The numpy array to save in .npy format.
            output_path: Destination path for the .npy file.
            metadata: Dict appended to the label's manifest.json.
        """
        temp_path = output_path.with_suffix(".tmp")
        np.save(str(temp_path), array)
        os.replace(str(temp_path) + ".npy", str(output_path))
        self._update_manifest(output_path.parent, metadata)

    def _update_manifest(self, label_directory: Path, entry: Dict) -> None:
        """Append one entry to the manifest.json in the label directory.

        The manifest is created if it does not exist.  It stores an array
        of sample metadata objects under the "samples" key.

        Args:
            label_directory: Directory containing the .npy samples.
            entry: Dict with file name, session ID, timestamp, etc.
        """
        manifest_path = label_directory / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        else:
            manifest = {
                "label": self.session.label,
                "gesture_type": entry.get("gesture_type", "unknown"),
                "samples": [],
            }
        manifest["samples"].append(entry)
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

    def _ensure_label_directories(self, label: str) -> None:
        """Create static and dynamic directories for the given label.

        Args:
            label: Gesture label string.
        """
        for gesture_type_name in ("static", "dynamic"):
            directory = self.config.data_root_directory / gesture_type_name / label
            directory.mkdir(parents=True, exist_ok=True)

    def _require_label(self) -> None:
        """Raise RuntimeError if no label has been set.

        Raises:
            RuntimeError: If self.session.label is None.
        """
        if self.session.label is None:
            raise RuntimeError(
                "No label set.  Press 'l' and enter a label before capturing."
            )

    def _prompt_label_via_terminal(self) -> None:
        """Block on terminal input to get the next label from the user.

        Pauses the OpenCV event loop during input (acceptable for a dev tool
        used by a single operator).
        """
        print("[collector] Enter label (then press Enter):")
        label = input().strip()
        if label:
            self.set_label(label)
        else:
            print("[collector] Empty input — label unchanged.")

    def _draw_overlay(
        self,
        frame: np.ndarray,
        current_landmark_frame: Optional[LandmarkFrame],
    ) -> None:
        """Draw the status bar and keyboard hints onto the frame.

        Args:
            frame: BGR frame to annotate in-place.
            current_landmark_frame: The latest landmark frame, or None if
                no hand is detected (used to show a "no hand" warning).
        """
        # Determine recording message.
        recording_message: Optional[str] = None
        if self._is_capturing_dynamic:
            elapsed = time.time() - self._dynamic_start_time
            remaining = max(0.0, self.config.dynamic_capture_duration_seconds - elapsed)
            recording_message = f"REC {remaining:.1f}s"

        draw_status_bar(
            frame,
            label=self.session.label,
            gesture_type_name=("DYNAMIC" if self._is_capturing_dynamic else "STATIC"),
            samples_captured=self.session.samples_captured,
            message=recording_message,
        )

        # Keyboard hint bar at the bottom.
        hint = "[l] label  [s] static  [d] dynamic start/stop  [q] quit"
        if current_landmark_frame is None:
            hint = "NO HAND DETECTED  |  " + hint
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            hint,
            (6, frame.shape[0] - 8),
            font,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _print_instructions() -> None:
        """Print keyboard shortcut instructions to the terminal."""
        print(
            "\n=== Nahual Collector ===\n"
            "  l  -- Enter label\n"
            "  s  -- Capture static sample\n"
            "  d  -- Start / stop dynamic capture\n"
            "  q  -- Quit\n"
            "========================\n"
        )
