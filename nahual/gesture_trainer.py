"""
nahual/gesture_trainer.py

Skeleton / placeholder for the LSM gesture classification model.

The model architecture (RandomForest, LSTM, CNN, etc.) is intentionally
left as TBD until real LSM data has been collected and inspected.  This
module defines a stable public interface so that all callers (main.py,
train.py, notebooks) can import and use GestureTrainer without needing
to change when the underlying algorithm is finalised.

All public methods contain ``raise NotImplementedError`` bodies.  Once
a training algorithm is chosen, only this module needs to be updated.

Data layout expected on disk (produced by GestureCollector):

    data/
      static/<label>/<uuid>.npy     # shape (21, 3), dtype float32
      dynamic/<label>/<uuid>.npy    # shape (N_frames, 21, 3), dtype float32
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Flat feature vector length for one static frame: 21 landmarks × 3 coords.
STATIC_FEATURE_LENGTH: int = 63

# Maximum number of frames per dynamic sample (mirrors gesture_heuristics.py).
MAX_DYNAMIC_FRAMES: int = 90

# Feature vector length for one frame in a dynamic sequence.
DYNAMIC_FRAME_FEATURE_LENGTH: int = STATIC_FEATURE_LENGTH


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for a training run.

    Attributes:
        data_root_directory: Path to the data/ directory created by
            GestureCollector.
        model_output_directory: Directory where trained model artifacts are saved.
        test_split_fraction: Fraction of samples reserved for the held-out
            test set.  Must be in (0, 1).
        random_seed: Integer seed for reproducibility across splits and
            model initialisation.
        model_type: String key that selects the algorithm.  Supported values
            will be defined when the training logic is implemented.
            Placeholder: "random_forest".
        model_hyperparameters: Algorithm-specific keyword arguments passed to
            the model constructor.  Empty dict uses defaults.
    """

    data_root_directory: Path = Path("data")
    model_output_directory: Path = Path("models")
    test_split_fraction: float = 0.2
    random_seed: int = 42
    model_type: str = "random_forest"
    model_hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Summary statistics from a completed training run.

    Attributes:
        accuracy: Overall accuracy on the held-out test set, in [0, 1].
        per_class_report: Dict mapping label → {precision, recall, f1, support}.
        confusion_matrix: numpy array of shape (N_classes, N_classes) where
            entry [i, j] is the number of samples with true label i predicted
            as label j.
        model_output_path: Path where the serialized model was written.
    """

    accuracy: float
    per_class_report: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    model_output_path: Path


# ---------------------------------------------------------------------------
# GestureTrainer
# ---------------------------------------------------------------------------


class GestureTrainer:
    """Loads collected data, trains a gesture classifier, and evaluates it.

    This class is a structured placeholder.  The public interface is fully
    specified so that callers can import and use it today; method bodies
    will be implemented once the algorithm is chosen.

    The API is deliberately model-agnostic: ``train`` will accept any
    estimator that follows a fit / predict interface, so a scikit-learn
    RandomForest, a PyTorch LSTM, or any other estimator can be used
    without changing the callers.

    Typical future usage::

        config = TrainingConfig(data_root_directory=Path("data"))
        trainer = GestureTrainer(config)

        feature_matrix, labels = trainer.load_static_data()
        result = trainer.train(feature_matrix, labels)
        print(f"Accuracy: {result.accuracy:.2%}")

        trainer.save_model()

    Args:
        config: TrainingConfig dataclass.  Defaults are applied if None.
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        """Initialise the trainer with optional configuration.

        Args:
            config: TrainingConfig instance.  Defaults are used if None.
        """
        self.config = config or TrainingConfig()
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_static_data(self) -> Tuple[np.ndarray, List[str]]:
        """Load all static gesture samples into a flat numpy matrix.

        Walks ``data/static/<label>/`` directories, loads each .npy file
        (shape (21, 3)), and flattens it to a 1-D vector of length 63.
        The label is inferred from the directory name.

        Returns:
            Tuple of:
                feature_matrix: numpy array of shape (N_samples, 63), dtype float32.
                labels: List of string labels, length N_samples, aligned
                    row-for-row with feature_matrix.

        Raises:
            FileNotFoundError: If the static data directory does not exist.
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError(
            "load_static_data is not yet implemented.  "
            "Collect training data with GestureCollector first."
        )

    def load_dynamic_data(self) -> Tuple[np.ndarray, List[str]]:
        """Load all dynamic gesture samples into a padded tensor.

        Walks ``data/dynamic/<label>/`` directories, loads each .npy file
        (shape (N_frames, 21, 3)), and pads or truncates to MAX_DYNAMIC_FRAMES
        so all samples have a uniform shape (90, 21, 3).  Each frame is then
        flattened to 63 values, giving a per-sample shape of (90, 63).

        Returns:
            Tuple of:
                sequence_tensor: numpy array of shape (N_samples, 90, 63),
                    dtype float32.
                labels: List of string labels, length N_samples.

        Raises:
            FileNotFoundError: If the dynamic data directory does not exist.
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError(
            "load_dynamic_data is not yet implemented.  "
            "Collect dynamic training data with GestureCollector first."
        )

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------

    def split_train_test(
        self,
        feature_matrix: np.ndarray,
        labels: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Stratified train / test split.

        Splits data while preserving the class distribution in both halves,
        using config.test_split_fraction and config.random_seed.

        Args:
            feature_matrix: numpy array of shape (N, F).
            labels: List of string labels, length N.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError

    def train(
        self,
        feature_matrix: np.ndarray,
        labels: List[str],
    ) -> TrainingResult:
        """Fit the model on training data and evaluate on the test split.

        Internally calls split_train_test, fits the model on the training
        partition, evaluates on the test partition, and calls save_model.

        Args:
            feature_matrix: numpy array of shape (N, F).
            labels: List of string labels, length N.

        Returns:
            TrainingResult containing accuracy, per-class metrics,
            confusion matrix, and the path of the saved model artifact.

        Raises:
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError

    def evaluate(
        self,
        feature_matrix: np.ndarray,
        labels: List[str],
    ) -> TrainingResult:
        """Evaluate a previously loaded model on arbitrary labelled data.

        Can be used to assess the trained model on a held-out validation set
        or on a fresh batch of samples collected after training.

        Args:
            feature_matrix: numpy array of shape (N, F).
            labels: List of string labels, length N.

        Returns:
            TrainingResult dataclass with evaluation statistics.

        Raises:
            RuntimeError: If no model has been loaded or trained yet.
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, output_path: Optional[Path] = None) -> Path:
        """Persist the trained model to disk.

        The serialisation format depends on config.model_type:
        - scikit-learn models: joblib pickle under models/<name>.pkl
        - PyTorch models: torch.save under models/<name>.pt

        Args:
            output_path: Override the default output path.  If None,
                the path is derived from config.model_output_directory
                and config.model_type.

        Returns:
            The path the model was saved to.

        Raises:
            RuntimeError: If no model has been trained yet.
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError

    def load_model(self, model_path: Path) -> None:
        """Deserialise a previously saved model into self._model.

        Args:
            model_path: Path to the serialised model file.

        Raises:
            FileNotFoundError: If model_path does not exist.
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feature_vector: np.ndarray) -> str:
        """Run inference on a single feature vector and return the label.

        This method is the bridge between the trained model and the real-time
        loop in main.py.  It accepts the same flat feature vector produced by
        GestureHeuristics to keep the inference path consistent with training.

        Args:
            feature_vector: For static gestures: numpy array of shape (63,).
                For dynamic gestures: numpy array of shape (90, 63).

        Returns:
            Predicted label string (e.g., "A", "B").

        Raises:
            RuntimeError: If no model has been loaded or trained.
            NotImplementedError: Until this method is implemented.
        """
        raise NotImplementedError
