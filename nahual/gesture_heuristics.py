"""
nahual/gesture_heuristics.py

Pure landmark preprocessing and feature extraction for LSM gesture detection.

This module is a data-transformation layer: it converts raw MediaPipe
HandLandmarkerResult objects into normalized numpy arrays and derived
features (angles, distances).  It has no I/O side effects, no mutable
global state, and is therefore safe to call from any thread.

All normalization is performed using hand_world_landmarks (metric,
hand-relative coordinates) rather than image-space landmarks, which
provides built-in scale invariance relative to the camera distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MediaPipe landmark indices.
WRIST_INDEX = 0
MIDDLE_FINGER_MCP_INDEX = 9  # Used as palm-size reference for normalization.

# Fingertip landmark indices in MediaPipe order.
FINGERTIP_INDICES: List[int] = [4, 8, 12, 16, 20]

# Per-finger joint triplets (proximal, middle, distal) used for angle computation.
# Each triplet is (parent_joint, pivot_joint, child_joint).
FINGER_JOINT_TRIPLETS: List[Tuple[int, int, int]] = [
    # Thumb
    (1, 2, 3),
    (2, 3, 4),
    # Index finger
    (5, 6, 7),
    (6, 7, 8),
    # Middle finger
    (9, 10, 11),
    (10, 11, 12),
    # Ring finger
    (13, 14, 15),
    (14, 15, 16),
    # Pinky
    (17, 18, 19),
    (18, 19, 20),
]

# Default landmark pairs for inter-landmark distance computation.
# Includes thumb-to-fingertip distances (useful for pinch/spread detection)
# and fingertip-to-wrist distances (useful for finger extension).
DEFAULT_LANDMARK_PAIRS: List[Tuple[int, int]] = [
    (4, 8),   # thumb tip  → index tip
    (4, 12),  # thumb tip  → middle tip
    (4, 16),  # thumb tip  → ring tip
    (4, 20),  # thumb tip  → pinky tip
    (0, 8),   # wrist      → index tip
    (0, 12),  # wrist      → middle tip
    (0, 16),  # wrist      → ring tip
    (0, 20),  # wrist      → pinky tip
]

# Velocity threshold (normalized units / ms) above which a gesture is
# classified as DYNAMIC.  This placeholder value will be refined once
# real LSM data is collected and analyzed.
MOTION_VELOCITY_THRESHOLD: float = 0.002

# Hard cap on sequence length for dynamic gestures.
# 90 frames ≈ 3 seconds at 30 fps.
MAX_DYNAMIC_FRAMES: int = 90


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class GestureType(Enum):
    """Discriminates between shape-only and motion-dependent gestures."""

    STATIC = auto()
    DYNAMIC = auto()


@dataclass
class LandmarkFrame:
    """One frame of hand landmark data.

    Attributes:
        coordinates: numpy array of shape (21, 3), dtype float32.
            Each row is [x, y, z] for one MediaPipe world landmark.
            Units are metres (metric, hand-relative).
        timestamp_ms: Frame timestamp in milliseconds from an arbitrary epoch.
    """

    coordinates: np.ndarray  # shape (21, 3), dtype float32
    timestamp_ms: int


@dataclass
class ExtractedFeatures:
    """All features derived from one or more LandmarkFrames.

    Attributes:
        normalized_coordinates: Wrist-centred, palm-width-scaled landmark
            positions. Shape (21, 3), dtype float32.
        finger_angles: Per-joint flexion angles in radians, derived from
            FINGER_JOINT_TRIPLETS. Shape (N_joints,), dtype float32.
        inter_landmark_distances: Euclidean distances between the landmark
            pairs defined in DEFAULT_LANDMARK_PAIRS (or a custom list).
            Shape (N_pairs,), dtype float32.
        gesture_type: Whether features represent a static or dynamic gesture.
        frame_sequence: For dynamic gestures, the full stacked sequence of
            normalized coordinates. Shape (N_frames, 21, 3), or None for
            static gestures.
    """

    normalized_coordinates: np.ndarray
    finger_angles: np.ndarray
    inter_landmark_distances: np.ndarray
    gesture_type: GestureType
    frame_sequence: Optional[np.ndarray] = None  # shape (N, 21, 3) or None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GestureHeuristics:
    """Rules-based preprocessing and feature extraction for hand landmarks.

    Provides a stable interface between raw MediaPipe output and the
    downstream collector, trainer, and real-time inference pipeline.
    Changing a heuristic (threshold, normalization strategy, feature set)
    only requires editing this class — no other module needs to change.

    Typical usage::

        heuristics = GestureHeuristics()
        frame = heuristics.extract_landmark_frame(mp_result, timestamp_ms=t)
        if frame is not None:
            features = heuristics.extract_features_static(frame)

    This class is intentionally stateless so that one instance can safely
    be shared across threads (e.g., main loop + collector UI thread).
    """

    def extract_landmark_frame(
        self,
        hand_landmarker_result,
        timestamp_ms: int,
        hand_index: int = 0,
    ) -> Optional[LandmarkFrame]:
        """Convert a MediaPipe result into a LandmarkFrame.

        Reads hand_world_landmarks (metric, hand-relative) rather than
        image-space hand_landmarks, providing scale invariance across
        different distances from the camera.

        Args:
            hand_landmarker_result: Result from HandLandmarker.detect_for_video.
            timestamp_ms: Frame timestamp in milliseconds.
            hand_index: Index of the hand to extract within the result.
                Always 0 under the single-hand constraint, but explicit for
                clarity and future flexibility.

        Returns:
            LandmarkFrame with a (21, 3) float32 coordinates array, or None
            if no hand was detected in the result.
        """
        world_landmarks = hand_landmarker_result.hand_world_landmarks
        if not world_landmarks or hand_index >= len(world_landmarks):
            return None

        hand = world_landmarks[hand_index]
        coordinates = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in hand],
            dtype=np.float32,
        )
        return LandmarkFrame(coordinates=coordinates, timestamp_ms=timestamp_ms)

    def normalize_coordinates(self, raw_coordinates: np.ndarray) -> np.ndarray:
        """Centre landmarks on the wrist and scale by palm width.

        Normalization steps:
            1. Translate: subtract the wrist position (landmark 0) so the
               wrist is at the origin.
            2. Scale: divide by the distance from wrist to middle-finger MCP
               (landmark 9), which approximates palm size and remains stable
               across different hand shapes and camera distances.

        This makes the resulting feature vectors invariant to hand position
        in 3-D space and to hand size.

        Args:
            raw_coordinates: numpy array of shape (21, 3), raw world coordinates.

        Returns:
            numpy array of shape (21, 3), dtype float32, with the wrist at the
            origin and landmarks scaled to palm-width units.
        """
        wrist_position = raw_coordinates[WRIST_INDEX]
        centred = raw_coordinates - wrist_position

        palm_width = float(
            np.linalg.norm(centred[MIDDLE_FINGER_MCP_INDEX] - centred[WRIST_INDEX])
        )
        if palm_width < 1e-6:
            # Degenerate case: avoid division by zero.
            return centred.astype(np.float32)

        return (centred / palm_width).astype(np.float32)

    def compute_finger_angles(self, normalized_coordinates: np.ndarray) -> np.ndarray:
        """Compute per-joint flexion angles using adjacent bone vectors.

        For each triplet (parent, pivot, child) in FINGER_JOINT_TRIPLETS,
        the angle at the pivot joint is the angle between the vector
        (pivot → parent) and the vector (pivot → child), giving a value
        in [0, π] radians.

        This angle encodes how bent each finger joint is, regardless of
        the overall orientation of the hand.

        Args:
            normalized_coordinates: numpy array of shape (21, 3).

        Returns:
            numpy array of shape (N_joints,), dtype float32, containing angles
            in radians.  N_joints equals len(FINGER_JOINT_TRIPLETS).
        """
        angles = []
        for parent_index, pivot_index, child_index in FINGER_JOINT_TRIPLETS:
            vector_to_parent = normalized_coordinates[parent_index] - normalized_coordinates[pivot_index]
            vector_to_child = normalized_coordinates[child_index] - normalized_coordinates[pivot_index]

            norm_parent = np.linalg.norm(vector_to_parent)
            norm_child = np.linalg.norm(vector_to_child)

            if norm_parent < 1e-6 or norm_child < 1e-6:
                angles.append(0.0)
                continue

            cosine = np.clip(
                np.dot(vector_to_parent, vector_to_child) / (norm_parent * norm_child),
                -1.0,
                1.0,
            )
            angles.append(float(np.arccos(cosine)))

        return np.array(angles, dtype=np.float32)

    def compute_inter_landmark_distances(
        self,
        normalized_coordinates: np.ndarray,
        landmark_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Compute Euclidean distances between specified landmark pairs.

        Useful for detecting pinch gestures (thumb-to-fingertip distances)
        and finger extension (wrist-to-fingertip distances).

        Args:
            normalized_coordinates: numpy array of shape (21, 3).
            landmark_pairs: List of (index_a, index_b) tuples.  If None,
                DEFAULT_LANDMARK_PAIRS is used.

        Returns:
            1-D numpy array of dtype float32, length = len(landmark_pairs).
        """
        pairs = landmark_pairs if landmark_pairs is not None else DEFAULT_LANDMARK_PAIRS
        distances = [
            float(np.linalg.norm(normalized_coordinates[index_a] - normalized_coordinates[index_b]))
            for index_a, index_b in pairs
        ]
        return np.array(distances, dtype=np.float32)

    def extract_features_static(self, landmark_frame: LandmarkFrame) -> ExtractedFeatures:
        """Build an ExtractedFeatures object from a single static frame.

        Normalizes coordinates, computes finger angles and inter-landmark
        distances.  Sets gesture_type to STATIC and frame_sequence to None.

        Args:
            landmark_frame: A LandmarkFrame produced by extract_landmark_frame.

        Returns:
            ExtractedFeatures with gesture_type=GestureType.STATIC.
        """
        normalized = self.normalize_coordinates(landmark_frame.coordinates)
        angles = self.compute_finger_angles(normalized)
        distances = self.compute_inter_landmark_distances(normalized)

        return ExtractedFeatures(
            normalized_coordinates=normalized,
            finger_angles=angles,
            inter_landmark_distances=distances,
            gesture_type=GestureType.STATIC,
            frame_sequence=None,
        )

    def extract_features_dynamic(
        self, landmark_frames: List[LandmarkFrame]
    ) -> ExtractedFeatures:
        """Build an ExtractedFeatures object from a sequence of frames.

        Normalizes each frame independently, then stacks them into a 3-D
        array of shape (N_frames, 21, 3).  Frame count is capped at
        MAX_DYNAMIC_FRAMES to enforce the 3-second limit.

        The normalized_coordinates, finger_angles, and inter_landmark_distances
        fields reflect the *last* frame of the sequence (most recent hand
        position), which is the natural choice for single-frame inference.

        Args:
            landmark_frames: Ordered list of LandmarkFrame objects (oldest first).

        Returns:
            ExtractedFeatures with gesture_type=GestureType.DYNAMIC and
            frame_sequence populated.

        Raises:
            ValueError: If landmark_frames is empty.
        """
        if not landmark_frames:
            raise ValueError("landmark_frames must not be empty.")

        capped_frames = landmark_frames[-MAX_DYNAMIC_FRAMES:]
        normalized_sequence = np.stack(
            [self.normalize_coordinates(frame.coordinates) for frame in capped_frames],
            axis=0,
        )  # shape (N_frames, 21, 3)

        # Derive scalar features from the final (most recent) frame.
        last_normalized = normalized_sequence[-1]
        angles = self.compute_finger_angles(last_normalized)
        distances = self.compute_inter_landmark_distances(last_normalized)

        return ExtractedFeatures(
            normalized_coordinates=last_normalized,
            finger_angles=angles,
            inter_landmark_distances=distances,
            gesture_type=GestureType.DYNAMIC,
            frame_sequence=normalized_sequence,
        )

    def classify_gesture_type(self, landmark_frames: List[LandmarkFrame]) -> GestureType:
        """Heuristically determine whether a gesture is static or dynamic.

        Computes the mean velocity of the wrist landmark across all frames.
        If the velocity exceeds MOTION_VELOCITY_THRESHOLD, the gesture is
        classified as DYNAMIC.

        This is a placeholder heuristic.  The threshold and the set of
        landmarks used will be refined once real LSM data is available.

        Args:
            landmark_frames: Two or more LandmarkFrame objects to analyze.
                A single frame always returns STATIC.

        Returns:
            GestureType.STATIC or GestureType.DYNAMIC.
        """
        if len(landmark_frames) < 2:
            return GestureType.STATIC

        total_displacement = 0.0
        total_time_ms = 0

        for previous_frame, current_frame in zip(landmark_frames[:-1], landmark_frames[1:]):
            delta_time_ms = current_frame.timestamp_ms - previous_frame.timestamp_ms
            if delta_time_ms <= 0:
                continue
            displacement = float(
                np.linalg.norm(
                    current_frame.coordinates[WRIST_INDEX]
                    - previous_frame.coordinates[WRIST_INDEX]
                )
            )
            total_displacement += displacement
            total_time_ms += delta_time_ms

        if total_time_ms == 0:
            return GestureType.STATIC

        mean_velocity = total_displacement / total_time_ms
        return GestureType.DYNAMIC if mean_velocity > MOTION_VELOCITY_THRESHOLD else GestureType.STATIC
