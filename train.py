"""
train.py

Entrypoint for the LSM gesture model trainer.

Loads collected samples from the data/ directory and trains a classifier.
This script is a placeholder — the training logic in GestureTrainer
will be implemented once data has been collected and the algorithm chosen.

Usage::

    uv run python train.py
"""

from pathlib import Path

from nahual.gesture_trainer import GestureTrainer, TrainingConfig


def main() -> None:
    """Train the LSM gesture classifier."""
    config = TrainingConfig(
        data_root_directory=Path("data"),
        model_output_directory=Path("models"),
    )
    trainer = GestureTrainer(config)

    print("Loading static gesture data ...")
    try:
        feature_matrix, labels = trainer.load_static_data()
        print(f"Loaded {len(labels)} static samples across {len(set(labels))} classes.")
        print("Training model ...")
        result = trainer.train(feature_matrix, labels)
        print(f"Training complete.  Accuracy: {result.accuracy:.2%}")
        print(f"Model saved to: {result.model_output_path}")
    except NotImplementedError:
        print(
            "GestureTrainer is not yet implemented.\n"
            "Collect training data with:\n"
            "    uv run python collect.py\n"
            "Then implement GestureTrainer in nahual/gesture_trainer.py."
        )


if __name__ == "__main__":
    main()
