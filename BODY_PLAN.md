# Body Plan

Plan for adding MediaPipe Pose based recognition ("body") next to the existing
hand landmark recognition ("hands"), delivered in small, reviewable steps
instead of one large refactor.

## Code names

* **hands** – The current pipeline: `HandLandmarker` (21 hand landmarks),
  static and dynamic letter classifiers. Its behavior and trained models must
  stay identical throughout this plan.
* **body** – The new pipeline under evaluation: `PoseLandmarker` (upper body,
  including 4 landmarks per hand: wrist, pinky, index, thumb). Hand-specific
  refinement for body is out of scope for now.

## Guiding principles

1. **One app, two pipelines.** Both run inside `main.py`; each has its own
   landmarker, heuristics and feature vectors.
1. **Share infrastructure, not features.** Training, visualization, the capture
   state machine and the collector UI are shared once there is a second
   consumer; feature extraction is never shared between hands and body.
1. **One-way dependencies.** `main.py` → `hands/`, `body/` → `core/`. `core/`
   never imports `hands/` or `body/`, and `hands/` and `body/` never import each
   other.
1. **Refactor only when a phase needs it.** No code is moved or generalized
   before the phase that consumes it.
1. **Hands stays byte-identical.** Any step that touches hands code is guarded
   by the parity check (see Phase 4.1).
1. **Existing scripts keep their interface.** `main.py`, `collect.py`,
   `train.py` and `inspect_data.py` keep their current flags and keys; new
   behavior is opt-in and defaults to hands.

## Glossary: pipeline

A **pipeline** is the full path from a camera frame to a prediction for one
kind of detection:

```
frame → landmarker → landmarks → normalize → feature vector → classifier → session → drawing
```

`hands` and `body` are the two pipelines. From Phase 4 on, `core/pipeline.py`
defines the contract (the steps every pipeline must provide), while
`hands/pipeline.py` and `body/pipeline.py` are the two implementations. Shared
code receives a pipeline object and calls its steps without knowing which one
it is. No pipeline files exist before Phase 4.

## Phases overview

| Phase | Goal                                                         | Touches hands code?                     |
| ----- | ------------------------------------------------------------ | --------------------------------------- |
| 1     | Define the changes needed for body detection (this document) | No                                      |
| 2     | Implement body detection only                                | No (only `main.py`, `visualization.py`) |
| 3     | Fine-tune body detection                                     | No                                      |
| 4     | Body heuristics and feature vectors for training             | Yes, guarded by parity check            |
| 5     | Collect body training samples                                | Yes (collector), guarded                |
| 6     | Run both together (recognition)                              | Yes (session), guarded                  |

Each phase, and each numbered step inside it, is meant to be one reviewable
change.

---

## Phase 1 – Define the changes

- [x] Code names, principles and phase breakdown agreed.
- [x] This plan reviewed.

---

## Phase 2 – Body detection only

**Goal:** see the pose skeleton live in `main.py`, with a key to switch between
hands only, body only, and both. No features, no training, no classifier.

| File                              | Status      | Change         | Why                                                                                                  |
| --------------------------------- | ----------- | -------------- | ---------------------------------------------------------------------------------------------------- |
| `models/pose_landmarker_full.task` | New (asset) | —             | MediaPipe pose model ("full" variant), next to `hand_landmarker.task`. The variant stays in the file name so it is clear which one body data was captured with. |
| `nahual/body/__init__.py`         | New         | Docstring only | Makes `body` a package so later body files have a home from day one.                                 |
| `nahual/body/landmarker.py`       | New         | ~100 lines     | `PoseLandmarkerConfig`, `build_pose_landmarker`, `detect_pose_landmarks`, same pattern as `hand_landmarker.py`. Separate file so body tuning can never alter the hand settings the hands models were trained under. |
| `nahual/visualization.py`         | Existing    | Small addition | `draw_pose_connections` using MediaPipe's `PoseLandmarksConnections`; mode label via the existing `draw_hint_bar`, stacked under the prediction panel (`draw_prediction_columns` now returns its height). |
| `main.py`                         | Existing    | Moderate       | Build both landmarkers at startup, add the `p` mode key, run hands and/or body per frame.            |
| `CLAUDE.md`, `nahual/__init__.py` | Existing    | Docs           | Mention `body/`, the pose asset, the `p` key; note that pose settings have no browser mirror yet.    |

**`main.py` behavior**

* Starts in **hands** mode, so launching it looks exactly like today.
* `p` cycles **hands → body → both**.
* `q`, `d`, `m` are unchanged; `d` and `m` act on the hands session only.
* Both landmarkers are built once and kept alive; the mode only decides which
  ones are called each frame (VIDEO mode tolerates skipped calls as long as
  timestamps increase).
* Re-entering a mode that includes hands creates a fresh
  `RealtimeGestureSession`, so no stale recording or motion state survives,
  with no change to `realtime_session.py`.
* In both-mode each landmarker converts the frame itself (two BGR→RGB
  conversions, ~1 ms). Accepted until `core/` exists.

**Untouched:** `gesture_heuristics.py`, `gesture_trainer.py`,
`realtime_session.py`, `gesture_collector.py`, `hand_landmarker.py`,
`collect.py`, `train.py`, `inspect_data.py`, `sign_language.py`, `web/`, all
data and models.

**Review checklist**

- [ ] Hands mode behaves exactly as before (same predictions, same keys).
- [ ] Body mode draws the pose skeleton with no hands overlay.
- [ ] Both mode draws both; FPS noted for each mode.
- [ ] Switching modes repeatedly causes no errors or stale hands predictions.

---

## Phase 3 – Fine-tune body detection

**Goal:** decide the detection settings body will be trained under.

* Tune `PoseLandmarkerConfig`: model variant (lite / full / heavy), detection,
  presence and tracking confidences.
* Decide which landmarks are relevant (expected: upper body 0–22; hips and
  legs are usually out of frame at a webcam).
* Optionally draw only the relevant landmarks.
* Measure FPS per mode and record the chosen values with a tuning note, as
  `HandLandmarkerConfig` does.

**Files:** `nahual/body/landmarker.py`, possibly `nahual/visualization.py`.

**Review checklist**

- [ ] Chosen config values documented with the reason for each.
- [ ] Landmark subset for body decided.

---

## Phase 4 – Body heuristics and feature vectors

**Goal:** body produces feature vectors the shared trainer can train on. This
is where the shared `core/` is introduced, one step at a time.

### 4.1 Parity check (before moving anything)

A small script that, for every sample under `data/<language>/`, records the
hands static vector, dynamic vector and the predictions of the current `.pkl`
models. Every later step that touches hands code must reproduce these exactly.

### 4.2 Extract the pure pieces into `core/`

| File                                | Status                                   | Change | Why                                                                                                  |
| ----------------------------------- | ---------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| `nahual/core/landmark_frame.py`     | New (split from `gesture_heuristics.py`) | Small  | `LandmarkFrame` and `interpolate_missing_frames` do not depend on the number of landmarks; body needs them. |
| `nahual/core/channel_statistics.py` | New (split from `gesture_heuristics.py`) | Small  | Per-channel temporal statistics over `(N_frames, C)`. The hands dynamic vector calls it and appends its hand-specific tail, keeping the 521 layout identical. |

`LandmarkFrame` is imported by `realtime_session.py`, `gesture_collector.py`
and `web/browser/session_bootstrap.py`; update those imports in the same step
and add `core/` to `NAHUAL_RUNTIME_MODULES` in `web/browser/build.py`.

- [ ] Parity check passes. Browser demo still loads and predicts.

### 4.3 Move hands into its own package

| File                         | Status                             | Change                 |
| ---------------------------- | ---------------------------------- | ---------------------- |
| `nahual/hands/landmarker.py` | Moved from `hand_landmarker.py`    | None                   |
| `nahual/hands/heuristics.py` | Moved from `gesture_heuristics.py` | What remains after 4.2 |

Update imports in the root scripts, `web/browser/build.py`
(`NAHUAL_RUNTIME_MODULES`), `web/browser/session_bootstrap.py` and
`CLAUDE.md`. Use `git mv` to keep history.

- [ ] Parity check passes. Browser demo still loads and predicts.

### 4.4 Generalize the trainer

| File                     | Status                          | Change   | Why                                                                                                  |
| ------------------------ | ------------------------------- | -------- | ---------------------------------------------------------------------------------------------------- |
| `nahual/core/trainer.py` | Moved from `gesture_trainer.py` | Moderate | Fit, evaluate, save, load and predict are already generic. Only the two loaders hard-code hand shapes; they take the expected shapes and the dynamic feature function from the pipeline. Saved `feature_length` metadata comes from the pipeline. |

Existing `.pkl` files contain only scikit-learn objects and plain metadata, so
moving modules does not break loading them.

- [ ] Parity check passes. `train.py` retrains hands with identical accuracy.

### 4.5 Pipeline contract and body features

| File                        | Status | Why                                                                                                  |
| --------------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| `nahual/core/pipeline.py`   | New    | The contract: detect, extract frame, normalize, static features, dynamic features, feature lengths, thresholds, skeleton. |
| `nahual/hands/pipeline.py`  | New    | Hands implementation, wrapping existing hands code.                                                  |
| `nahual/body/heuristics.py` | New    | Body normalization (body-relative frame) and its own static and dynamic vectors.                     |
| `nahual/body/pipeline.py`   | New    | Body implementation.                                                                                 |

Paths gain a pipeline level (`data/hands/<language>/`, `data/body/<language>/`,
`models/hands/<language>/`, `models/body/<language>/`) via
`sign_language.py`; existing hands data and models are moved with `git mv`.

- [ ] Parity check passes.
- [ ] Body vectors have the expected fixed length on recorded frames.

---

## Phase 5 – Collect body samples

**Goal:** capture body training data without any risk of mixing it with hands
data.

| File                                               | Status                            | Change   | Why                                                                                                  |
| -------------------------------------------------- | --------------------------------- | -------- | ---------------------------------------------------------------------------------------------------- |
| `nahual/core/collector.py`                         | Moved from `gesture_collector.py` | Moderate | The UI and keys (`l`, `s`, `d`, `q`) stay the same; it saves whatever the selected pipeline produces. |
| `collect.py`                                       | Existing                          | Small    | Pipeline selection flag, defaulting to hands. One pipeline per run, never both.                      |
| `nahual/core/data_inspector.py`, `inspect_data.py` | Moved / existing                  | Minor    | Read shapes from the files instead of hard-coding `(21, 3)`; pipeline flag.                          |
| `train.py`                                         | Existing                          | Small    | Pipeline selection flag, defaulting to hands.                                                        |

- [ ] `collect.py` with no new flag behaves exactly as today.
- [ ] Body samples land only under `data/body/<language>/`.
- [ ] First body classifier trains.

---

## Phase 6 – Both together

**Goal:** body recognition runs live next to hands.

| File                                                   | Status                           | Change    | Why                                                                                     |
| ------------------------------------------------------ | -------------------------------- | --------- | --------------------------------------------------------------------------------------- |
| `nahual/core/capture_session.py`                       | Moved from `realtime_session.py` | **Major** | See below.                                                                              |
| `nahual/visualization.py` (or `core/visualization.py`) | Existing                         | Small     | Prediction panel takes a label and slot so hands and body panels can be drawn together. |
| `main.py`                                              | Existing                         | Moderate  | One session per active pipeline; `d` / `m` scoped per the decision below.               |

**Why the session change is major.** The state machine itself (idle,
recording, still-frame and timeout detection, latching) stays the same, but
`realtime_session.py` is hand-specific in six places today:

1. It imports `GestureHeuristics` directly.
2. `process_frame(landmark_frame, handedness)` mirrors left hands.
3. The motion signal uses the hand `normalize_coordinates`.
4. Static prediction uses the hand 81-value vector.
5. Dynamic classification uses the hand 521-value vector.
6. Thresholds are module constants tuned for letters.

Items 1–5 become pipeline calls and item 6 moves into each pipeline. It is the
live recognition core for both the desktop app and the browser demo, so it is
done last and guarded by the parity check plus a replay of a recorded session.

- [ ] Parity check passes; browser demo unchanged.
- [ ] Decide how `d` and `m` behave in both-mode.

---

## Out of scope for now

* Body in the browser demo (needs the JavaScript `PoseLandmarker` and a
  mirrored config in `web/browser/app.js`).
* Full hand detail inside body (two `HandLandmarker` hands, handedness
  tracking).
* Continuous signing segmentation and sequence models.
* Packaging `nahual` as an installable dependency.