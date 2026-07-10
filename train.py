"""
train.py

Entrypoint for the LSM gesture model trainer.

Loads collected samples from the data/ directory and trains the static and
dynamic gesture classifiers, saving each trained model to the models/
directory.

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

    # --- Static gesture model -------------------------------------------
    print("Loading static gesture data ...")
    try:
        feature_matrix, labels = trainer.load_static_data()
        print(
            f"Loaded {len(labels)} static samples "
            f"across {len(set(labels))} classes."
        )
        print("Training static model ...")
        result = trainer.train(feature_matrix, labels)
        print(f"Static training complete.  Accuracy: {result.accuracy:.2%}")
        print(f"Static model saved to: {result.model_output_path}")
    except FileNotFoundError:
        print(
            "No static data found.  Collect training data with:\n"
            "    uv run python collect.py"
        )
    except ValueError as error:
        print(f"Static training skipped: {error}")

    # --- Dynamic gesture model ------------------------------------------
    print("\nLoading dynamic gesture data ...")
    try:
        dynamic_feature_matrix, dynamic_labels = trainer.load_dynamic_data()
        print(
            f"Loaded {len(dynamic_labels)} dynamic samples "
            f"across {len(set(dynamic_labels))} classes."
        )
        print("Training dynamic model ...")
        dynamic_result = trainer.train_dynamic(dynamic_feature_matrix, dynamic_labels)
        print(
            f"Dynamic training complete.  " f"Accuracy: {dynamic_result.accuracy:.2%}"
        )
        print(f"Dynamic model saved to: {dynamic_result.model_output_path}")
    except FileNotFoundError:
        print(
            "No dynamic data found — skipping dynamic model training.\n"
            "Collect dynamic samples with:\n"
            "    uv run python collect.py  (press 'd' to record)"
        )
    except ValueError as error:
        print(f"Dynamic training skipped: {error}")


if __name__ == "__main__":
    main()
