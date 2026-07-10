# AGENTS Guidelines for This Repository

This repository contains a Python application. When working on the project interactively
with an agent please follow the guidelines below so that the development experience
continues to work smoothly.

This application's purpose is to detect custom gestures with machine learning.
The custom gestures are LSM (Lengua de Señas Mexicana), a sign language for
México. The scope is to recognize every letter of the LSM alphabet, split
across two classifiers (static poses and dynamic motion gestures), usable from
a desktop OpenCV demo.

# Project Structure

* `nahual/` – Core inference and machine learning package.
  * `gesture_collector.py` – Interactive webcam data collection tool.
  * `gesture_heuristics.py` – Landmark preprocessing and feature extraction.
  * `gesture_trainer.py` – Model training and inference (static + dynamic).
  * `visualization.py` – OpenCV drawing helpers (landmarks, overlays).
  * `data_inspector.py` – Dataset inspection utilities (sample counts per label).
* Root entry scripts:
  * `main.py` – Real-time desktop demo (webcam + OpenCV window).
  * `collect.py` – Run the interactive data collector.
  * `train.py` – Train the static and dynamic classifiers.
  * `inspect_data.py` – Print a table of sample counts per label.
* `data/` – Collected `.npy` samples grouped by label (see gesture types below).
* `models/` – Trained artifacts: `gesture_classifier.pkl` (static),
  `dynamic_gesture_classifier.pkl` (dynamic), and the MediaPipe
  `hand_landmarker.task` model asset.

## Running the Project

All scripts are launched through `uv`:

* `uv run python main.py` – Desktop real-time recognition demo.
* `uv run python collect.py` – Collect labeled gesture samples.
* `uv run python train.py` – Train the classifiers from `data/`.
* `uv run python inspect_data.py` – Inspect dataset sample counts.

# Architecture Notes

* Real-time recognition runs in `main.py`, which feeds MediaPipe hand landmarks
  through `GestureHeuristics` (feature extraction) and `GestureTrainer`
  (static + dynamic inference).
* Feature extraction is centralized in `nahual/gesture_heuristics.py`; the
  collector (`gesture_collector.py`) and the demo (`main.py`) share the same
  helpers (e.g. `GestureHeuristics.flatten_static_features`) so the vectors used
  for training and inference stay identical.

# Static vs. Dynamic Gestures

The LSM alphabet is split into two gesture types, each with its own dataset
folder and its own trained model:

* **Static gestures** – Held hand poses with no motion. Stored in
  `data/static/`, trained into `models/gesture_classifier.pkl`.
  Labels: `letra_a`, `letra_b`, `letra_c`, `letra_d`, `letra_e`, `letra_f`,
  `letra_g`, `letra_h`, `letra_i`, `letra_l`, `letra_m`, `letra_n`, `letra_o`,
  `letra_p`, `letra_r`, `letra_s`, `letra_t`, `letra_u`, `letra_v`, `letra_w`,
  `letra_y`.
* **Dynamic gestures** – Letters that require hand movement, captured as short
  sequences. Stored in `data/dynamic/`, trained into
  `models/dynamic_gesture_classifier.pkl`.
  Labels: `letra_j`, `letra_k`, `letra_q`, `letra_x`, `letra_z`, `letra_ñ`.

# Keep Dependencies in Sync

* If you add or update dependencies remember to run `uv lock` to update the
  lockfile.
* `pyproject.toml` / `uv.lock` define the full environment for the desktop tools
  and development (includes `mediapipe`, `opencv-python`, `black`, `isort`).

# Version Constraints

* The project targets Python `>=3.9,<3.13` (see `pyproject.toml`).
* The trained `.pkl` models are pickled with specific versions
  (Python 3.9, `scikit-learn==1.6.1`, `numpy==2.0.2`) from the environment used
  to train them. Bumping these versions or retraining under a different
  environment can break unpickling — change them deliberately and regenerate the
  models when you do.

# Reasoning Process

* Always reason step-by-step
* Validate feasibility before proposing scaling solutions
* Explain trade-offs explicitly
* Provide rationale for architectural choices

## Coding Conventions

* Use PEP 8 – Style Guide for Python Code for coding conventions
* Comment every function with its purpose, arguments and a quick explanation of the function
* Do not abbreviate variables, use full name for better readability

## Commit instructions

* Use conventional commits for messages: a `<type>` prefix (`fix:`, `feat:`,
  `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`)
  followed by the commit message.
* Formatting with `black` and `isort` is applied automatically to edited
  Python files via a `PostToolUse` hook (see `.claude/settings.json`), so no
  manual formatting step is needed before committing.

