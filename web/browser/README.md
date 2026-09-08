# Nahual — browser-only demo (`web/browser/`)

A self-contained, **server-less** version of the Nahual LSM demo. MediaPipe runs
in the browser (as before), but the recognition — feature extraction, the motion
state machine, and both Random Forest classifiers — runs **client-side under
[Pyodide](https://pyodide.org)** (WebAssembly Python) using the project's *real*
`nahual` code. Nothing is reimplemented in JavaScript.

Because there is no network in the per-frame loop, the loop is fully synchronous
(detect → recognise → draw, in order, every frame) exactly like `main.py`, so no
frames are dropped and the dynamic-gesture buffer fills at the camera rate. This
is what fixes the lag/frame-loss of the old FastAPI server demo.

## Files

| File | Role |
|---|---|
| `index.html` | Page markup + loads the Pyodide runtime from CDN |
| `app.js` | MediaPipe in JS, the synchronous render loop, prediction bars |
| `session_bootstrap.py` | Runs inside Pyodide: loads the models, creates the session, exposes `process_frame_js` / `toggle_manual_js` / `status_js` |
| `style.css` | Styles (self-contained copy) |
| `build.py` | Assembles a deployable `web/dist/` bundle |

At runtime the page also needs the real `nahual/*.py` source and the trained
LSM models (`models/lsm/*.pkl` plus the shared `models/hand_landmarker.task`)
— `build.py` copies those into a flat `dist/models/` so `nahual/` stays the
single source of truth. The desktop tools support ASL too, but this demo
bundles LSM only.

## Run it locally

**Option A — straight from the repo (no build):**

```bash
python3 -m http.server 8000        # from the repo root
# then open http://localhost:8000/web/browser/index.html
```

**Option B — the packaged bundle (what you deploy):**

```bash
python3 web/browser/build.py
python3 -m http.server --directory web/dist 8000
# then open http://localhost:8000
```

Use `http://localhost` (or HTTPS) — browsers block camera access on plain-http
LAN IPs. First load downloads the Pyodide runtime + scikit-learn from a CDN
(one-time, then cached); the loading overlay shows progress.

## Deploy it (for partners to click a link)

1. `python3 web/browser/build.py` → produces `web/dist/` (~14 MB, self-contained).
2. Upload `web/dist/` to any **static** host (Netlify, Cloudflare Pages, GitHub
   Pages, a Render *Static Site*, etc.). Its publish/root directory = `web/dist`.
3. That's it — no server, no runtime, nothing to keep warm or maintain. Static
   hosts serve over HTTPS, so the camera works.

The host must serve `.py` and `.pkl` as ordinary static files (all the common
hosts do).

## Notes

- **Keep MediaPipe settings in sync.** `initialiseHandLandmarker()` in `app.js`
  mirrors `HandLandmarkerConfig` in `nahual/hand_landmarker.py`. If you change
  the Python config, update this too (same rule as the old web demo).
- **Confidence gate.** The dynamic gate `DYNAMIC_CONFIDENCE_THRESHOLD` in
  `nahual/realtime_session.py` is currently `0.0` (a diagnostic that latches
  every prediction). For the build you share with partners, set it back to
  `0.65` so idle hand-drift doesn't surface as random letters.
- **Debug hook.** `window.__nahual` (in `app.js`) exposes `processFrame`,
  `toggleManual`, `status`, `runPython`, and `diagnostics` for testing without a
  camera. Harmless (client-side only); remove it if you want a leaner page.
- **No retrain needed.** Pyodide ships newer scikit-learn (1.7.x) than the models
  were pickled with (1.6.1); predictions were verified bit-identical to the
  desktop build, so the existing `.pkl` files work as-is.
