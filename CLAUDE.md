# AGENTS Guidelines for This Repository

This repository contains a Python application. When working on the project interactively
with an agent please follow the guidelines below so that the development experience
continues to work smoothly.

This application's purpose is to detect custom gestures with machine learning.
It supports multiple sign languages — currently LSM (Lengua de Señas
Mexicana) and ASL (American Sign Language) — each with its own dataset and
trained classifiers, selected at runtime with the `-lsm` / `-asl` flag
(default `-lsm`). For each language, the scope is to recognize every letter of its
alphabet, split across two classifiers (static poses and dynamic motion
gestures), usable from a desktop OpenCV demo.

# Project Structure

* `nahual/` – Core inference and machine learning package.
  * `gesture_collector.py` – Interactive webcam data collection tool.
  * `hand_landmarker.py` – Canonical MediaPipe HandLandmarker configuration
    (`HandLandmarkerConfig`) and helpers (`build_hand_landmarker`,
    `detect_landmarks`) shared by the desktop tools.
  * `gesture_heuristics.py` – Landmark preprocessing and feature extraction.
  * `gesture_trainer.py` – Model training and inference (static + dynamic).
  * `sign_language.py` – Supported language codes (`SIGN_LANGUAGES`), the
    shared `-lsm` / `-asl` argparse helper (`parse_sign_language_argument`),
    and the per-language path helpers (`data_directory`, `models_directory`)
    used by all four root scripts. Not imported by any module bundled into
    the browser demo (see `web/browser/build.py`'s `NAHUAL_RUNTIME_MODULES`).
  * `visualization.py` – OpenCV drawing helpers (landmarks, overlays).
  * `data_inspector.py` – Dataset inspection utilities (sample counts per label).
* Root entry scripts (each accepts `-lsm` or `-asl`, default `-lsm`):
  * `main.py` – Real-time desktop demo (webcam + OpenCV window).
  * `collect.py` – Run the interactive data collector.
  * `train.py` – Train the static and dynamic classifiers.
  * `inspect_data.py` – Print a table of sample counts per label.
* `data/<language>/` – Collected `.npy` samples grouped by label, under a
  per-language directory (`data/lsm/`, `data/asl/`); see gesture types below.
* `models/` – Trained artifacts. `hand_landmarker.task` (the MediaPipe model
  asset) is language-independent and lives at this top level.
  `<language>/gesture_classifier.pkl` (static) and
  `<language>/dynamic_gesture_classifier.pkl` (dynamic) are per-language
  (e.g. `models/lsm/gesture_classifier.pkl`).

## Running the Project

All scripts are launched through `uv` and default to LSM; pass `-asl` to
operate on the ASL dataset/models instead.

* `uv run python main.py` – Desktop real-time recognition demo.
* `uv run python collect.py` – Collect labeled gesture samples.
* `uv run python train.py` – Train the classifiers from `data/<language>/`.
* `uv run python inspect_data.py` – Inspect dataset sample counts.

# Architecture Notes

* Real-time recognition runs in `main.py`, which feeds MediaPipe hand landmarks
  through `GestureHeuristics` (feature extraction) and `GestureTrainer`
  (static + dynamic inference).
* Feature extraction is centralized in `nahual/gesture_heuristics.py`; the
  collector (`gesture_collector.py`) and the demo (`main.py`) share the same
  helpers (e.g. `GestureHeuristics.flatten_static_features`) so the vectors used
  for training and inference stay identical.
* MediaPipe hand-detection settings (VIDEO mode, `num_hands`, the detection /
  presence / tracking confidences) are centralized in
  `nahual/hand_landmarker.py` as `HandLandmarkerConfig`. `main.py` and
  `gesture_collector.py` both build their landmarker from it, so the collector
  captures training data under the exact settings the demo recognizes it with.
  **The browser demo (`web/browser/app.js`) cannot import Python, so it keeps a
  manual copy of these same values in `initialiseHandLandmarker`. Whenever you
  change `HandLandmarkerConfig`, update the mirrored options in
  `web/browser/app.js` to match (and vice versa) — the two must stay in sync to
  avoid different detection behavior across the desktop and web front-ends.**

# Static vs. Dynamic Gestures

Each sign language's alphabet is split into two gesture types, each with its
own dataset folder and its own trained model. Labels use a `<category>_<value>`
naming scheme (currently only the `letter_` category exists); the language
itself is **not** encoded in the label, since the `data/<language>/` and
`models/<language>/` directory level already carries it.

## LSM (Lengua de Señas Mexicana)

* **Static gestures** – Held hand poses with no motion. Stored in
  `data/lsm/static/`, trained into `models/lsm/gesture_classifier.pkl`.
  Labels: `letter_a`, `letter_b`, `letter_c`, `letter_d`, `letter_e`,
  `letter_f`, `letter_g`, `letter_h`, `letter_i`, `letter_l`, `letter_m`,
  `letter_n`, `letter_o`, `letter_p`, `letter_r`, `letter_s`, `letter_t`,
  `letter_u`, `letter_v`, `letter_w`, `letter_y`.
* **Dynamic gestures** – Letters that require hand movement, captured as short
  sequences. Stored in `data/lsm/dynamic/`, trained into
  `models/lsm/dynamic_gesture_classifier.pkl`.
  Labels: `letter_j`, `letter_k`, `letter_q`, `letter_x`, `letter_z`, `letter_ñ`.

## ASL (American Sign Language)

* **Static gestures** – Stored in `data/asl/static/`, trained into
  `models/asl/gesture_classifier.pkl`. Planned labels: every letter except J
  and Z (`letter_a`–`letter_y`, excluding `letter_j` and `letter_z`). ASL has
  no Ñ.
* **Dynamic gestures** – Stored in `data/asl/dynamic/`, trained into
  `models/asl/dynamic_gesture_classifier.pkl`. Planned labels: `letter_j`,
  `letter_z`.

As of this writing the ASL directories and models do not yet exist — they are
created on first `uv run python collect.py -asl` /
`uv run python train.py -asl`.

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

