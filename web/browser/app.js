/*
 * web/browser/app.js
 *
 * Browser-only Nahual demo — no server.
 *
 * MediaPipe HandLandmarker runs locally in JavaScript to obtain hand
 * landmarks; those landmarks are fed straight into the project's real Python
 * recognition code (RealtimeGestureSession) running client-side under Pyodide
 * (WebAssembly Python). Because there is no network in the per-frame loop, the
 * loop is fully synchronous — detect -> recognise -> draw, in order, every
 * animation frame — exactly like the desktop main.py. No frames are dropped
 * waiting on a server, so the dynamic gesture buffer fills at the camera rate.
 *
 * Nothing about the recognition logic lives here: this file only runs
 * MediaPipe, draws the skeleton, forwards landmarks into Pyodide, and renders
 * the prediction bars from the overlay Python returns.
 */

import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/vision_bundle.mjs";

const MEDIAPIPE_WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm";
const HAND_LANDMARKER_MODEL_PATH = "/models/hand_landmarker.task";

// Where the nahual Python source and trained models are fetched from (served
// at the site root) and where they are placed inside the Pyodide filesystem.
const NAHUAL_SOURCE_BASE = "/nahual";
const MODELS_BASE = "/models";
const PYODIDE_SESSION_DIR = "/session";
const NAHUAL_PYTHON_FILES = [
  "__init__.py",
  "gesture_heuristics.py",
  "gesture_trainer.py",
  "realtime_session.py",
];
const MODEL_PICKLES = [
  "gesture_classifier.pkl",
  "dynamic_gesture_classifier.pkl",
];

const LOW_CONFIDENCE_THRESHOLD = 0.65; // Mirrors visualization.py.

// Standard MediaPipe hand skeleton topology (landmark index pairs).
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [5, 9], [9, 10], [10, 11], [11, 12], // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17], // palm base
];

// Per-landmark colors, indexed by MediaPipe landmark id (0..20). These mirror
// LANDMARK_NAMES in nahual/visualization.py.
const PALM = "#EF4838";
const LANDMARK_COLORS = [
  PALM, PALM, "#FBE6B6", "#FBE6B6", "#FBE6B6",
  PALM, "#7A447F", "#7A447F", "#7A447F",
  PALM, "#F8CE27", "#F8CE27", "#F8CE27",
  PALM, "#70FA3B", "#70FA3B", "#70FA3B",
  PALM, "#2D64BE", "#2D64BE", "#2D64BE",
];

const videoElement = document.getElementById("webcam");
const overlayCanvas = document.getElementById("overlay-canvas");
const overlayContext = overlayCanvas.getContext("2d");
const staticBar = document.getElementById("static-bar");
const staticLabel = document.getElementById("static-label");
const staticSecondary = document.getElementById("static-secondary");
// Low-confidence warning disabled for now (kept for later re-enable); see the
// commented span in index.html and the commented line in updateStaticBar.
// const staticWarning = document.getElementById("static-warning");
const recordingBar = document.getElementById("recording-bar");
const recordingInfo = document.getElementById("recording-info");
const dynamicBar = document.getElementById("dynamic-bar");
const dynamicLabel = document.getElementById("dynamic-label");
const dynamicSecondary = document.getElementById("dynamic-secondary");
const recordButton = document.getElementById("record-button");
const statusMessage = document.getElementById("status-message");
const motionDebugCheckbox = document.getElementById("motion-debug-checkbox");
const motionDebugReadout = document.getElementById("motion-debug-readout");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");

let handLandmarker = null;
let pyodide = null;
// PyProxy handles to the Python functions defined in session_bootstrap.py.
let processFrameFn = null;
let toggleManualFn = null;

let lastVideoTimestamp = -1;
let motionStartThreshold = 0.015;
let motionStopThreshold = 0.008;
let dynamicModelAvailable = false;
let isRunning = false;
let pyodideReady = false;

// --- Effective-FPS instrumentation (the browser twin of main.py's readout).
// EMA-smoothed processed frame rate plus the per-stage split between MediaPipe
// detection and the Pyodide recognition call, so "no frame loss" and per-frame
// cost can be verified live and during validation.
const FPS_SMOOTHING_ALPHA = 0.1;
let previousFrameTime = -1;
let smoothedFps = 0;
let smoothedDetectMs = 0;
let smoothedRecogniseMs = 0;

/**
 * Update the status line shown to the user.
 * @param {string} message Text to display.
 */
function setStatus(message) {
  statusMessage.textContent = message;
}

/**
 * Update the full-screen loading overlay text while Pyodide warms up.
 * @param {string} message Progress text.
 */
function setLoading(message) {
  if (loadingText) {
    loadingText.textContent = message;
  }
}

/**
 * Strip the "letra_" prefix from a label for display.
 * @param {string} label Raw model label (e.g. "letra_a").
 * @returns {string} Display letter (e.g. "a").
 */
function displayLetter(label) {
  return label.startsWith("letra_") ? label.slice("letra_".length) : label;
}

/**
 * Initialise the MediaPipe HandLandmarker in VIDEO mode.
 *
 * Mirrors the canonical detection settings defined in Python in
 * nahual/hand_landmarker.py (HandLandmarkerConfig). The browser cannot import
 * Python, so these values are a manual mirror — keep them in sync with
 * HandLandmarkerConfig so the demo detects hands identically to how the
 * training data was collected.
 */
async function initialiseHandLandmarker() {
  const visionFileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_PATH);
  handLandmarker = await HandLandmarker.createFromOptions(visionFileset, {
    baseOptions: { modelAssetPath: HAND_LANDMARKER_MODEL_PATH },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
}

/**
 * Load Pyodide, install scikit-learn, copy the nahual package + trained models
 * into the Pyodide filesystem, and run the bootstrap that creates the session.
 *
 * All of this is one-time startup cost (cached by the browser afterwards). The
 * function resolves once the Python side is ready to process frames.
 */
async function initialisePyodide() {
  setLoading("Loading Python runtime…");
  // loadPyodide is provided by the classic <script> tag in index.html.
  pyodide = await loadPyodide();

  setLoading("Loading scikit-learn (one-time)…");
  await pyodide.loadPackage(["scikit-learn"]);

  setLoading("Loading recognition code…");
  pyodide.FS.mkdirTree(`${PYODIDE_SESSION_DIR}/nahual`);
  pyodide.FS.mkdirTree(`${PYODIDE_SESSION_DIR}/models`);

  // Copy the real nahual Python source (single source of truth — not a JS
  // reimplementation) into the Pyodide filesystem.
  for (const fileName of NAHUAL_PYTHON_FILES) {
    const source = await fetchText(`${NAHUAL_SOURCE_BASE}/${fileName}`);
    pyodide.FS.writeFile(`${PYODIDE_SESSION_DIR}/nahual/${fileName}`, source);
  }

  setLoading("Loading trained models…");
  for (const modelName of MODEL_PICKLES) {
    const bytes = await fetchBytes(`${MODELS_BASE}/${modelName}`);
    pyodide.FS.writeFile(`${PYODIDE_SESSION_DIR}/models/${modelName}`, bytes);
  }

  setLoading("Starting recognition session…");
  pyodide.runPython(`import sys; sys.path.insert(0, "${PYODIDE_SESSION_DIR}")`);
  const bootstrapSource = await fetchText("./session_bootstrap.py");
  pyodide.runPython(bootstrapSource);

  processFrameFn = pyodide.globals.get("process_frame_js");
  toggleManualFn = pyodide.globals.get("toggle_manual_js");

  const status = JSON.parse(pyodide.globals.get("status_js")());
  dynamicModelAvailable = status.dynamic_model_available;
  if (typeof status.motion_start_threshold === "number") {
    motionStartThreshold = status.motion_start_threshold;
  }
  if (typeof status.motion_stop_threshold === "number") {
    motionStopThreshold = status.motion_stop_threshold;
  }
  if (!status.static_model_available && !dynamicModelAvailable) {
    setStatus("Warning: no trained models were loaded.");
  }
  pyodideReady = true;

  // Validation/debug hook: lets the recognition path be exercised without a
  // live camera (synthetic landmark payloads) and exposes the in-browser
  // Python for parity checks against the desktop build.
  window.__nahual = {
    ready: () => pyodideReady,
    status: () => JSON.parse(pyodide.globals.get("status_js")()),
    processFrame: (payload) => JSON.parse(processFrameFn(JSON.stringify(payload))),
    toggleManual: () => toggleManualFn && toggleManualFn(),
    runPython: (code) => pyodide.runPython(code),
    diagnostics: () => window.__nahualDiagnostics || null,
  };
}

// Local dev wants fresh files on every reload; production wants the browser and
// Render's CDN to cache them normally. Forcing "no-store" in production makes
// every startup fetch bypass the CDN and hit the origin, which can briefly 404
// a file that exists right around a deploy — so only use it on localhost.
const IS_LOCALHOST = ["localhost", "127.0.0.1"].includes(location.hostname);
const FETCH_CACHE_MODE = IS_LOCALHOST ? "no-store" : "default";

/**
 * Fetch a URL with a few retries and linear backoff.
 *
 * The static bundle is served from a CDN; immediately around a deploy (or on an
 * edge cache miss) a request can transiently 404 a file that does exist. A
 * single plain fetch would then abort startup, so retry a couple of times to
 * ride over that brief window.
 * @param {string} url URL to fetch.
 * @param {number} retries Number of attempts before giving up.
 * @returns {Promise<Response>} An OK response.
 */
async function fetchWithRetry(url, retries = 6) {
  let lastStatus = 0;
  for (let attempt = 0; attempt < retries; attempt += 1) {
    try {
      const response = await fetch(url, { cache: FETCH_CACHE_MODE });
      if (response.ok) {
        return response;
      }
      lastStatus = response.status;
    } catch (error) {
      lastStatus = -1; // Network-level failure (e.g. dropped connection).
    }
    if (attempt < retries - 1) {
      // Linear backoff: 300ms, 600ms, 900ms, ...
      await new Promise((resolve) => setTimeout(resolve, 300 * (attempt + 1)));
    }
  }
  throw new Error(`fetch ${url} -> HTTP ${lastStatus} after ${retries} attempts`);
}

/**
 * Fetch a URL as text (with retries).
 * @param {string} url URL to fetch.
 * @returns {Promise<string>} Response body as text.
 */
async function fetchText(url) {
  return (await fetchWithRetry(url)).text();
}

/**
 * Fetch a URL as bytes (with retries).
 * @param {string} url URL to fetch.
 * @returns {Promise<Uint8Array>} Response body as bytes.
 */
async function fetchBytes(url) {
  return new Uint8Array(await (await fetchWithRetry(url)).arrayBuffer());
}

/**
 * Convert MediaPipe worldLandmarks objects into a plain [x, y, z] array.
 * @param {Array<{x:number,y:number,z:number}>} worldLandmarks One hand's metric landmarks.
 * @returns {number[][]} Array of 21 [x, y, z] triples.
 */
function toCoordinateArray(worldLandmarks) {
  return worldLandmarks.map((point) => [point.x, point.y, point.z]);
}

/**
 * Feed one frame's landmarks into the Pyodide session and return the overlay.
 *
 * Runs synchronously: the Python recognition executes inline, so the returned
 * overlay describes exactly this frame (no stale server state). The payload
 * crosses into Python as a JSON string to avoid per-frame proxy management.
 * @param {object|null} detection MediaPipe detection result for this frame.
 * @returns {object} The overlay dict from RealtimeGestureSession.process_frame.
 */
function recogniseFrame(detection) {
  const hasHand =
    detection && detection.worldLandmarks && detection.worldLandmarks.length > 0;

  const payload = {
    landmarks: hasHand ? toCoordinateArray(detection.worldLandmarks[0]) : null,
    handedness:
      hasHand && detection.handedness && detection.handedness.length > 0
        ? detection.handedness[0][0].categoryName
        : null,
    timestamp_ms: Math.round(performance.now()),
  };

  const overlayJson = processFrameFn(JSON.stringify(payload));
  return JSON.parse(overlayJson);
}

/**
 * Draw one hand's skeleton (connections + landmark dots) on the overlay
 * canvas, or clear it when no hand is visible. Colors mirror
 * nahual/visualization.py so the web overlay matches the desktop one.
 * @param {object|null} detection MediaPipe detection result for this frame.
 */
function drawSkeleton(detection) {
  const width = overlayCanvas.width;
  const height = overlayCanvas.height;
  overlayContext.clearRect(0, 0, width, height);

  if (!detection || !detection.landmarks || detection.landmarks.length === 0) {
    return;
  }
  const landmarks = detection.landmarks[0];

  overlayContext.lineWidth = 3;
  for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    overlayContext.strokeStyle =
      LANDMARK_COLORS[Math.max(startIndex, endIndex)] || "#ffffff";
    overlayContext.beginPath();
    overlayContext.moveTo(start.x * width, start.y * height);
    overlayContext.lineTo(end.x * width, end.y * height);
    overlayContext.stroke();
  }

  for (let index = 0; index < landmarks.length; index += 1) {
    const point = landmarks[index];
    overlayContext.fillStyle = LANDMARK_COLORS[index] || "#ffffff";
    overlayContext.beginPath();
    overlayContext.arc(point.x * width, point.y * height, 4, 0, Math.PI * 2);
    overlayContext.fill();
  }
}

/**
 * Update the static column. The "Static:" header and the background stay put;
 * only the detected letter and the Hand/Confidence line appear or clear.
 * @param {object} overlay Overlay for the latest processed frame.
 */
function updateStaticBar(overlay) {
  if (!overlay.static_label) {
    // Idle: keep the "Static:" header and background; clear the value + line.
    staticLabel.textContent = "Static:";
    staticSecondary.textContent = "";
    return;
  }
  staticLabel.textContent = `Static: ${displayLetter(overlay.static_label)}`;

  const confidencePercent = (overlay.static_confidence * 100).toFixed(0);
  staticSecondary.textContent = overlay.handedness
    ? `Hand: ${overlay.handedness} | Confidence: ${confidencePercent}%`
    : `Confidence: ${confidencePercent}%`;
  // Low-confidence warning disabled for now (kept for later re-enable):
  // staticWarning.hidden = overlay.static_confidence >= LOW_CONFIDENCE_THRESHOLD;
}

/**
 * Update the recording indicator (row 2). Its background stays put; the text
 * shows only while a recording is active. The "RECORDING" label was dropped to
 * cut visual noise. Manual recordings have no countdown, so seconds are omitted.
 * @param {object} overlay Overlay for the latest processed frame.
 */
function updateRecordingBar(overlay) {
  if (overlay.capture_state !== "RECORDING") {
    // Idle: keep the background; clear the text.
    recordingInfo.textContent = "";
    return;
  }
  recordingInfo.textContent = overlay.manual_capture
    ? `manual  |  ${overlay.buffer_length} frames`
    : `auto  |  ${overlay.recording_remaining_seconds.toFixed(1)}s remaining` +
      `  |  ${overlay.buffer_length} frames`;
}

/**
 * Update the dynamic column. The "Dynamic:" header and background stay put; the
 * detected letter and the Confidence line appear while a result is latched (the
 * Python session owns the 3s display window) and clear afterwards.
 * @param {object} overlay Overlay for the latest processed frame.
 */
function updateDynamicBar(overlay) {
  if (!overlay.dynamic_label) {
    // Idle: keep the "Dynamic:" header and background; clear the value + line.
    dynamicLabel.textContent = "Dynamic:";
    dynamicSecondary.textContent = "";
    return;
  }
  dynamicLabel.textContent = `Dynamic: ${displayLetter(overlay.dynamic_label)}`;
  dynamicSecondary.textContent =
    `Confidence: ${(overlay.dynamic_confidence * 100).toFixed(0)}%`;
  // Frame count disabled for now (kept for later re-enable):
  //   + `  |  ${overlay.dynamic_frame_count} frames`;
}

/**
 * Keep the manual record button label in sync with the recording state.
 * @param {object} overlay Overlay for the latest processed frame.
 */
function updateRecordButton(overlay) {
  recordButton.textContent =
    overlay.capture_state === "RECORDING"
      ? "Stop recording & classify"
      : "Start manual recording";
}

/**
 * Update the motion-debug readout (the web twin of the desktop 'm' overlay).
 * Adds fps and the per-stage timing split so no-frame-loss and per-frame cost
 * can be watched live. Hidden unless the toggle is on.
 * @param {object} overlay Overlay for the latest processed frame.
 */
function updateMotionDebug(overlay) {
  if (!motionDebugCheckbox.checked) {
    motionDebugReadout.hidden = true;
    return;
  }
  motionDebugReadout.hidden = false;
  motionDebugReadout.textContent = [
    `raw motion : ${overlay.raw_motion.toFixed(4)}`,
    `smoothed   : ${overlay.smoothed_motion.toFixed(4)}`,
    `start/stop : ${motionStartThreshold.toFixed(4)}/` +
      `${motionStopThreshold.toFixed(4)}`,
    `state      : ${overlay.capture_state}  buf=${overlay.buffer_length}`,
    `fps        : ${smoothedFps.toFixed(1)}`,
    `ms det/rec : ${smoothedDetectMs.toFixed(1)}/${smoothedRecogniseMs.toFixed(1)}`,
  ].join("\n");
}

/**
 * The render loop: detect locally, draw the skeleton, run the Python
 * recogniser inline, and refresh the prediction bars once per animation frame.
 *
 * Every detected frame is recognised before the next is requested, so the
 * dynamic buffer fills at the true camera rate — there is no send/response gate
 * and therefore no frame loss.
 */
function renderLoop() {
  if (!isRunning) {
    return;
  }

  if (videoElement.readyState >= 2) {
    if (overlayCanvas.width !== videoElement.videoWidth) {
      overlayCanvas.width = videoElement.videoWidth;
      overlayCanvas.height = videoElement.videoHeight;
    }

    let detection = null;
    const timestamp = performance.now();
    // MediaPipe requires strictly increasing timestamps.
    if (timestamp > lastVideoTimestamp) {
      lastVideoTimestamp = timestamp;
      const detectStart = performance.now();
      detection = handLandmarker.detectForVideo(videoElement, timestamp);
      const detectMs = performance.now() - detectStart;

      drawSkeleton(detection);

      if (pyodideReady) {
        const recogniseStart = performance.now();
        const overlay = recogniseFrame(detection);
        const recogniseMs = performance.now() - recogniseStart;

        updateStaticBar(overlay);
        updateRecordingBar(overlay);
        updateRecordButton(overlay);
        updateDynamicBar(overlay);
        updateMotionDebug(overlay);

        updateTimings(timestamp, detectMs, recogniseMs);
      }
    } else {
      drawSkeleton(detection);
    }
  }

  window.requestAnimationFrame(renderLoop);
}

/**
 * Update the EMA-smoothed fps and per-stage timing diagnostics.
 * @param {number} frameTime performance.now() for this frame.
 * @param {number} detectMs Milliseconds spent in MediaPipe detection.
 * @param {number} recogniseMs Milliseconds spent in the Pyodide recogniser.
 */
function updateTimings(frameTime, detectMs, recogniseMs) {
  if (previousFrameTime >= 0) {
    const deltaSeconds = (frameTime - previousFrameTime) / 1000;
    if (deltaSeconds > 0) {
      const instantaneousFps = 1 / deltaSeconds;
      smoothedFps =
        FPS_SMOOTHING_ALPHA * instantaneousFps +
        (1 - FPS_SMOOTHING_ALPHA) * smoothedFps;
    }
  }
  previousFrameTime = frameTime;
  smoothedDetectMs =
    FPS_SMOOTHING_ALPHA * detectMs + (1 - FPS_SMOOTHING_ALPHA) * smoothedDetectMs;
  smoothedRecogniseMs =
    FPS_SMOOTHING_ALPHA * recogniseMs +
    (1 - FPS_SMOOTHING_ALPHA) * smoothedRecogniseMs;
  // Expose for automated validation.
  window.__nahualDiagnostics = {
    fps: smoothedFps,
    detectMs: smoothedDetectMs,
    recogniseMs: smoothedRecogniseMs,
  };
}

/**
 * Bootstrap: guard the secure context, start MediaPipe and Pyodide in
 * parallel, start the camera, and enter the render loop.
 */
async function bootstrap() {
  // Browsers only expose the camera API on secure origins (HTTPS or
  // localhost). Explain the fix instead of a generic "denied" message.
  if (!window.isSecureContext || !navigator.mediaDevices) {
    setLoading(
      "Camera blocked: open this page via http://localhost or HTTPS, " +
        `not ${location.hostname}.`
    );
    return;
  }

  try {
    // MediaPipe and Pyodide are independent, heavy loads — warm them together.
    setLoading("Loading hand tracking + Python runtime…");
    await Promise.all([initialiseHandLandmarker(), initialisePyodide()]);
  } catch (error) {
    setLoading(`Startup failed: ${error.message}`);
    return;
  }

  let mediaStream = null;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 } },
      audio: false,
    });
  } catch (error) {
    setLoading("Camera access denied or unavailable. Allow it and reload.");
    return;
  }

  videoElement.srcObject = mediaStream;
  await new Promise((resolve) => {
    videoElement.onloadedmetadata = resolve;
  });

  recordButton.disabled = !dynamicModelAvailable;
  setStatus("Ready. Show a sign to the camera.");
  if (loadingOverlay) {
    loadingOverlay.hidden = true;
  }

  isRunning = true;
  window.requestAnimationFrame(renderLoop);
}

recordButton.addEventListener("click", () => {
  if (toggleManualFn) {
    toggleManualFn();
  }
});

bootstrap();
