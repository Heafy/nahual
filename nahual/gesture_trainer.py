"""
nahual/gesture_trainer.py

LSM gesture classification trainer using Random Forest (Phase 1).

Loads collected static gesture samples from disk, trains a
RandomForestClassifier from scikit-learn, evaluates it on a held-out
test split, and persists the trained model for real-time inference.

The public interface is model-agnostic so that callers (main.py,
train.py) do not need to change when the underlying algorithm evolves
in future phases.

Data layout expected on disk (produced by GestureCollector):

    data/
      static/<label>/<uuid>.npy     # shape (81,), dtype float32
                                    #   [0:63]  normalized_coordinates (21×3)
                                    #   [63:73] finger_angles (10 joints)
                                    #   [73:81] inter_landmark_distances (8 pairs)
      dynamic/<label>/<uuid>.npy    # shape (N_frames, 21, 3), dtype float32
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from nahual.gesture_heuristics import (DYNAMIC_STATISTICAL_FEATURE_LENGTH,
                                       GestureHeuristics)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Flat feature vector length for one static frame:
#   63 normalized coordinates (21 landmarks × 3 axes)
#   + 10 finger joint angles
#   + 8 inter-landmark distances
STATIC_FEATURE_LENGTH: int = 81

# Maximum number of frames per dynamic sample (mirrors gesture_heuristics.py).
MAX_DYNAMIC_FRAMES: int = 90

# Feature vector length for one frame in a dynamic sequence.
DYNAMIC_FRAME_FEATURE_LENGTH: int = STATIC_FEATURE_LENGTH

# Minimum number of samples per class before a warning is logged.
MINIMUM_SAMPLES_PER_CLASS_WARNING: int = 10

# Default hyperparameters for RandomForestClassifier, tuned for the
# typical LSM alphabet dataset (~81 features, small sample counts).
DEFAULT_RF_HYPERPARAMETERS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
}


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
        model_type: String key that selects the algorithm.  Currently
            supported: "random_forest".
        model_hyperparameters: Algorithm-specific keyword arguments passed to
            the model constructor.  Empty dict uses defaults defined in
            DEFAULT_RF_HYPERPARAMETERS.
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

    Uses scikit-learn's RandomForestClassifier for Phase 1 (static gesture
    classification).  The API is deliberately model-agnostic so that future
    phases can swap in different estimators without changing callers.

    Typical usage::

        config = TrainingConfig(data_root_directory=Path("data"))
        trainer = GestureTrainer(config)

        feature_matrix, labels = trainer.load_static_data()
        result = trainer.train(feature_matrix, labels)
        print(f"Accuracy: {result.accuracy:.2%}")

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
        self._label_encoder: Optional[LabelEncoder] = None
        self._dynamic_model: Optional[Any] = None
        self._dynamic_label_encoder: Optional[LabelEncoder] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_static_data(self) -> Tuple[np.ndarray, List[str]]:
        """Load all static gesture samples into a flat numpy matrix.

        Walks ``data/static/<label>/`` directories and loads each .npy file
        (shape (81,), dtype float32).  The label is inferred from the parent
        directory name.

        Each feature vector contains 63 normalised landmark coordinates,
        10 finger joint angles, and 8 inter-landmark distances concatenated
        into a single flat array.

        Returns:
            Tuple of:
                feature_matrix: numpy array of shape (N_samples, 81),
                    dtype float32.
                labels: List of string labels, length N_samples, aligned
                    row-for-row with feature_matrix.

        Raises:
            FileNotFoundError: If the static data directory does not exist.
            ValueError: If no valid .npy sample files are found.
        """
        static_directory = self.config.data_root_directory / "static"
        if not static_directory.exists():
            raise FileNotFoundError(
                f"Static data directory not found: {static_directory}.  "
                "Collect training data with GestureCollector first."
            )

        feature_vectors: List[np.ndarray] = []
        labels: List[str] = []
        samples_per_class: Dict[str, int] = {}

        for label_directory in sorted(static_directory.iterdir()):
            if not label_directory.is_dir():
                continue

            label = label_directory.name
            sample_count = 0

            for sample_file in sorted(label_directory.glob("*.npy")):
                sample = np.load(str(sample_file))

                if sample.shape != (STATIC_FEATURE_LENGTH,):
                    logger.warning(
                        "Skipping %s: expected shape (%d,), got %s.",
                        sample_file,
                        STATIC_FEATURE_LENGTH,
                        sample.shape,
                    )
                    continue

                feature_vectors.append(sample)
                labels.append(label)
                sample_count += 1

            samples_per_class[label] = sample_count

        # Warn about classes with too few samples.
        for label, count in samples_per_class.items():
            if count < MINIMUM_SAMPLES_PER_CLASS_WARNING:
                logger.warning(
                    "Class '%s' has only %d samples (minimum recommended: %d).  "
                    "Consider collecting more data for this class.",
                    label,
                    count,
                    MINIMUM_SAMPLES_PER_CLASS_WARNING,
                )

        if not feature_vectors:
            raise ValueError(
                f"No valid .npy sample files found in {static_directory}.  "
                "Collect training data with GestureCollector first."
            )

        feature_matrix = np.stack(feature_vectors).astype(np.float32)
        logger.info(
            "Loaded %d static samples across %d classes: %s",
            len(labels),
            len(samples_per_class),
            samples_per_class,
        )
        return feature_matrix, labels

    def load_dynamic_data(self) -> Tuple[np.ndarray, List[str]]:
        """Load all dynamic gesture samples and extract statistical features.

        Walks ``data/dynamic/<label>/`` directories, loads each .npy file
        (shape (N_frames, 21, 3)), and computes temporal statistical features
        via GestureHeuristics.extract_statistical_features_dynamic() to produce
        a fixed-length feature vector per sample.

        No padding is needed because the statistical feature extraction
        inherently handles variable-length sequences.

        Returns:
            Tuple of:
                feature_matrix: numpy array of shape
                    (N_samples, DYNAMIC_STATISTICAL_FEATURE_LENGTH),
                    dtype float32.
                labels: List of string labels, length N_samples.

        Raises:
            FileNotFoundError: If the dynamic data directory does not exist.
            ValueError: If no valid .npy sample files are found.
        """
        dynamic_directory = self.config.data_root_directory / "dynamic"
        if not dynamic_directory.exists():
            raise FileNotFoundError(
                f"Dynamic data directory not found: {dynamic_directory}.  "
                "Collect dynamic training data with GestureCollector first."
            )

        heuristics = GestureHeuristics()
        feature_vectors: List[np.ndarray] = []
        labels: List[str] = []
        samples_per_class: Dict[str, int] = {}

        for label_directory in sorted(dynamic_directory.iterdir()):
            if not label_directory.is_dir():
                continue

            label = label_directory.name
            class_sample_count = 0

            for sample_path in sorted(label_directory.glob("*.npy")):
                try:
                    sample = np.load(sample_path)
                except Exception:
                    logger.warning("Failed to load %s — skipping.", sample_path)
                    continue

                if sample.ndim != 3 or sample.shape[1:] != (21, 3):
                    logger.warning(
                        "Unexpected shape %s in %s — expected (N, 21, 3).  "
                        "Skipping.",
                        sample.shape,
                        sample_path,
                    )
                    continue

                statistical_features = heuristics.extract_statistical_features_dynamic(
                    sample
                )
                feature_vectors.append(statistical_features)
                labels.append(label)
                class_sample_count += 1

            if class_sample_count > 0:
                samples_per_class[label] = class_sample_count
                if class_sample_count < MINIMUM_SAMPLES_PER_CLASS_WARNING:
                    logger.warning(
                        "Class '%s' has only %d dynamic samples.  "
                        "Consider collecting at least %d for reliable training.",
                        label,
                        class_sample_count,
                        MINIMUM_SAMPLES_PER_CLASS_WARNING,
                    )

        if not feature_vectors:
            raise ValueError(
                f"No valid .npy sample files found in {dynamic_directory}.  "
                "Collect dynamic training data with GestureCollector first."
            )

        feature_matrix = np.stack(feature_vectors).astype(np.float32)
        logger.info(
            "Loaded %d dynamic samples across %d classes: %s",
            len(labels),
            len(samples_per_class),
            samples_per_class,
        )
        return feature_matrix, labels

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
        using config.test_split_fraction and config.random_seed.  Falls back
        to a non-stratified split if any class has too few samples for
        stratification.

        Args:
            feature_matrix: numpy array of shape (N, F).
            labels: List of string labels, length N.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) where X values are
            numpy arrays and y values are lists of string labels.
        """
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                feature_matrix,
                labels,
                test_size=self.config.test_split_fraction,
                random_state=self.config.random_seed,
                stratify=labels,
            )
        except ValueError:
            logger.warning(
                "Stratified split failed (a class may have too few samples).  "
                "Falling back to a non-stratified split."
            )
            x_train, x_test, y_train, y_test = train_test_split(
                feature_matrix,
                labels,
                test_size=self.config.test_split_fraction,
                random_state=self.config.random_seed,
            )

        logger.info(
            "Split data: %d training samples, %d test samples.",
            len(y_train),
            len(y_test),
        )
        return x_train, x_test, y_train, y_test

    def train(
        self,
        feature_matrix: np.ndarray,
        labels: List[str],
    ) -> TrainingResult:
        """Fit the model on training data and evaluate on the test split.

        Internally calls split_train_test, fits a RandomForestClassifier on
        the training partition, evaluates on the test partition, and persists
        the model to disk via save_model.

        The Random Forest is configured with DEFAULT_RF_HYPERPARAMETERS merged
        with any user-provided overrides in config.model_hyperparameters.

        Args:
            feature_matrix: numpy array of shape (N, F).
            labels: List of string labels, length N.

        Returns:
            TrainingResult containing accuracy, per-class metrics,
            confusion matrix, and the path of the saved model artifact.
        """
        # Fit the label encoder on the full label set so all classes are known.
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(labels)

        # Build the Random Forest with sensible defaults, allowing user overrides.
        hyperparameters = {
            **DEFAULT_RF_HYPERPARAMETERS,
            **self.config.model_hyperparameters,
        }
        hyperparameters["random_state"] = self.config.random_seed
        self._model = RandomForestClassifier(**hyperparameters)

        logger.info(
            "Training RandomForestClassifier with hyperparameters: %s",
            hyperparameters,
        )

        # Split, fit, evaluate.
        x_train, x_test, y_train, y_test = self.split_train_test(feature_matrix, labels)
        self._model.fit(x_train, y_train)

        # Evaluate on the held-out test set.
        result = self.evaluate(x_test, y_test)

        # Persist the trained model.
        saved_path = self.save_model()
        result.model_output_path = saved_path

        logger.info("Training complete.  Test accuracy: %.2f%%", result.accuracy * 100)
        return result

    def evaluate(
        self,
        feature_matrix: np.ndarray,
        labels: List[str],
    ) -> TrainingResult:
        """Evaluate a previously loaded model on arbitrary labelled data.

        Computes accuracy, per-class precision / recall / f1, and a confusion
        matrix against the provided ground-truth labels.

        Args:
            feature_matrix: numpy array of shape (N, F).
            labels: List of string labels, length N.

        Returns:
            TrainingResult dataclass with evaluation statistics.

        Raises:
            RuntimeError: If no model has been loaded or trained yet.
        """
        if self._model is None:
            raise RuntimeError(
                "No model available.  Train a model or load one from disk first."
            )

        predictions = self._model.predict(feature_matrix)

        accuracy = accuracy_score(labels, predictions)

        # Build per-class report, extracting only the actual label entries.
        full_report = classification_report(
            labels, predictions, output_dict=True, zero_division=0
        )
        known_labels = (
            list(self._label_encoder.classes_)
            if self._label_encoder is not None
            else sorted(set(labels))
        )
        per_class_report = {
            label: full_report[label] for label in known_labels if label in full_report
        }

        # Confusion matrix with consistent row/column ordering.
        confusion = confusion_matrix(labels, predictions, labels=known_labels)

        return TrainingResult(
            accuracy=accuracy,
            per_class_report=per_class_report,
            confusion_matrix=confusion,
            model_output_path=Path(""),
        )

    # ------------------------------------------------------------------
    # Dynamic gesture training
    # ------------------------------------------------------------------

    def train_dynamic(
        self,
        feature_matrix: np.ndarray,
        labels: List[str],
    ) -> TrainingResult:
        """Fit a dynamic gesture model and evaluate on the test split.

        Mirrors train() but operates on the dynamic model attributes
        (_dynamic_model, _dynamic_label_encoder) so both static and
        dynamic models can coexist within the same GestureTrainer instance.

        Args:
            feature_matrix: numpy array of shape
                (N, DYNAMIC_STATISTICAL_FEATURE_LENGTH).
            labels: List of string labels, length N.

        Returns:
            TrainingResult containing accuracy, per-class metrics,
            confusion matrix, and the path of the saved model artifact.
        """
        self._dynamic_label_encoder = LabelEncoder()
        self._dynamic_label_encoder.fit(labels)

        hyperparameters = {
            **DEFAULT_RF_HYPERPARAMETERS,
            **self.config.model_hyperparameters,
        }
        hyperparameters["random_state"] = self.config.random_seed
        self._dynamic_model = RandomForestClassifier(**hyperparameters)

        logger.info(
            "Training dynamic RandomForestClassifier with hyperparameters: %s",
            hyperparameters,
        )

        x_train, x_test, y_train, y_test = self.split_train_test(feature_matrix, labels)
        self._dynamic_model.fit(x_train, y_train)

        # Evaluate using the dynamic model temporarily swapped into _model
        # so that the existing evaluate() method works unchanged.
        original_model = self._model
        original_encoder = self._label_encoder
        self._model = self._dynamic_model
        self._label_encoder = self._dynamic_label_encoder

        result = self.evaluate(x_test, y_test)

        # Restore the static model references.
        self._model = original_model
        self._label_encoder = original_encoder

        saved_path = self.save_dynamic_model()
        result.model_output_path = saved_path

        logger.info(
            "Dynamic training complete.  Test accuracy: %.2f%%",
            result.accuracy * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, output_path: Optional[Path] = None) -> Path:
        """Persist the trained model to disk using joblib.

        Serialises a dictionary containing the model, label encoder, and
        metadata so that load_model can fully restore the trainer state.

        Args:
            output_path: Override the default output path.  If None,
                the path is derived from config.model_output_directory
                as ``models/gesture_classifier.pkl``.

        Returns:
            The path the model was saved to.

        Raises:
            RuntimeError: If no model has been trained yet.
        """
        if self._model is None:
            raise RuntimeError("No model to save.  Train a model first.")

        if output_path is None:
            output_path = self.config.model_output_directory / "gesture_classifier.pkl"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self._model,
            "label_encoder": self._label_encoder,
            "feature_length": STATIC_FEATURE_LENGTH,
            "model_type": self.config.model_type,
        }
        joblib.dump(artifact, output_path)
        logger.info("Model saved to %s", output_path)
        return output_path

    def load_model(self, model_path: Path) -> None:
        """Deserialise a previously saved model into self._model.

        Restores both the trained estimator and the label encoder from a
        joblib artifact produced by save_model.

        Args:
            model_path: Path to the serialised model file (.pkl).

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        artifact = joblib.load(model_path)
        self._model = artifact["model"]
        self._label_encoder = artifact["label_encoder"]

        logger.info(
            "Loaded %s model from %s with %d classes.",
            artifact.get("model_type", "unknown"),
            model_path,
            len(self._label_encoder.classes_) if self._label_encoder else 0,
        )

    def save_dynamic_model(self, output_path: Optional[Path] = None) -> Path:
        """Persist the trained dynamic gesture model to disk using joblib.

        Serialises a dictionary containing the dynamic model, label encoder,
        and metadata so that load_dynamic_model can restore the state.

        Args:
            output_path: Override the default output path.  If None,
                the path is derived from config.model_output_directory
                as ``models/dynamic_gesture_classifier.pkl``.

        Returns:
            The path the model was saved to.

        Raises:
            RuntimeError: If no dynamic model has been trained yet.
        """
        if self._dynamic_model is None:
            raise RuntimeError(
                "No dynamic model to save.  Train a dynamic model first."
            )

        if output_path is None:
            output_path = (
                self.config.model_output_directory / "dynamic_gesture_classifier.pkl"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self._dynamic_model,
            "label_encoder": self._dynamic_label_encoder,
            "feature_length": DYNAMIC_STATISTICAL_FEATURE_LENGTH,
            "model_type": "random_forest_dynamic",
        }
        joblib.dump(artifact, output_path)
        logger.info("Dynamic model saved to %s", output_path)
        return output_path

    def load_dynamic_model(self, model_path: Path) -> None:
        """Deserialise a previously saved dynamic model.

        Restores the dynamic estimator and label encoder from a joblib
        artifact produced by save_dynamic_model.

        Args:
            model_path: Path to the serialised dynamic model file (.pkl).

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Dynamic model file not found: {model_path}")

        artifact = joblib.load(model_path)
        self._dynamic_model = artifact["model"]
        self._dynamic_label_encoder = artifact["label_encoder"]

        logger.info(
            "Loaded %s model from %s with %d classes.",
            artifact.get("model_type", "unknown"),
            model_path,
            (
                len(self._dynamic_label_encoder.classes_)
                if self._dynamic_label_encoder
                else 0
            ),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    # TODO: Check if this methods is still used
    def predict(self, feature_vector: np.ndarray) -> str:
        """Run inference on a single feature vector and return the label.

        This method is the bridge between the trained model and the real-time
        loop in main.py.  It accepts the same flat feature vector produced by
        GestureHeuristics to keep the inference path consistent with training.

        Args:
            feature_vector: numpy array of shape (81,) for static gestures.

        Returns:
            Predicted label string (e.g., "letra_a", "letra_b").

        Raises:
            RuntimeError: If no model has been loaded or trained.
        """
        if self._model is None:
            raise RuntimeError(
                "No model available.  Train a model or load one from disk first."
            )

        reshaped_vector = feature_vector.reshape(1, -1)
        prediction = self._model.predict(reshaped_vector)
        return prediction[0]

    def predict_with_confidence(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """Run inference and return the predicted label together with its confidence.

        Uses the classifier's class probability estimates (predict_proba) to
        derive a confidence score: the probability assigned to the winning class.

        Args:
            feature_vector: numpy array of shape (81,) for static gestures.

        Returns:
            A tuple of (label, confidence) where label is the predicted class
            string and confidence is a float in [0, 1].

        Raises:
            RuntimeError: If no model has been loaded or trained.
        """
        if self._model is None:
            raise RuntimeError(
                "No model available.  Train a model or load one from disk first."
            )

        reshaped_vector = feature_vector.reshape(1, -1)
        probabilities = self._model.predict_proba(reshaped_vector)[0]
        predicted_index = int(probabilities.argmax())
        label = self._model.classes_[predicted_index]
        confidence = float(probabilities[predicted_index])
        return label, confidence

    def predict_dynamic_with_confidence(
        self, feature_vector: np.ndarray
    ) -> tuple[str, float]:
        """Run inference on a dynamic gesture statistical feature vector.

        Uses the dynamic model's class probability estimates to predict
        the gesture label and associated confidence score.

        Args:
            feature_vector: numpy array of shape
                (DYNAMIC_STATISTICAL_FEATURE_LENGTH,) produced by
                GestureHeuristics.extract_statistical_features_dynamic().

        Returns:
            A tuple of (label, confidence) where label is the predicted
            class string and confidence is a float in [0, 1].

        Raises:
            RuntimeError: If no dynamic model has been loaded or trained.
        """
        if self._dynamic_model is None:
            raise RuntimeError(
                "No dynamic model available.  "
                "Train a dynamic model or load one from disk first."
            )

        reshaped_vector = feature_vector.reshape(1, -1)
        probabilities = self._dynamic_model.predict_proba(reshaped_vector)[0]
        predicted_index = int(probabilities.argmax())
        label = self._dynamic_model.classes_[predicted_index]
        confidence = float(probabilities[predicted_index])
        return label, confidence
