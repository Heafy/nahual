/*
 * web/static/app.js
 *
 * Browser side of the Nahual detection demo.
 *
 * Responsibilities (deliberately thin — no gesture logic lives here):
 *   1. Run MediaPipe HandLandmarker locally to obtain hand landmarks.
 *   2. Draw the hand skeleton on the overlay canvas every animation frame.
 *   3. Stream the metric `worldLandmarks` (tiny JSON, never video) over a
 *      WebSocket to the FastAPI server, which runs the shared
 *      RealtimeGestureSession for all feature extraction and classification.
 *   4. Render the prediction bars (S = static, RECORDING, D = dynamic) from
 *      the overlay the server returns.
 *
 * Sends are response-gated (one in flight at a time) while drawing runs at
 * full requestAnimationFrame rate, so the video and skeleton stay live even
 * if the server lags.
 */

import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/vision_bundle.mjs";

const MEDIAPIPE_WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm";
const HAND_LANDMARKER_MODEL_PATH = "/models/hand_landmarker.task";

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

// Per-landmark colors, indexed by MediaPipe landmark id (0..20).
// These mirror LANDMARK_NAMES in nahual/visualization.py (palm/MCP joints
// are gray-red; each finger's distal joints share that finger's color).
const PALM = "#EF4838";
const LANDMARK_COLORS = [
  PALM, // 0  WRIST
  PALM, // 1  THUMB_CMC
  "#FBE6B6", // 2  THUMB_MCP
  "#FBE6B6", // 3  THUMB_IP
  "#FBE6B6", // 4  THUMB_TIP
  PALM, // 5  INDEX_FINGER_MCP
  "#7A447F", // 6  INDEX_FINGER_PIP
  "#7A447F", // 7  INDEX_FINGER_DIP
  "#7A447F", // 8  INDEX_FINGER_TIP
  PALM, // 9  MIDDLE_FINGER_MCP
  "#F8CE27", // 10 MIDDLE_FINGER_PIP
  "#F8CE27", // 11 MIDDLE_FINGER_DIP
  "#F8CE27", // 12 MIDDLE_FINGER_TIP
  PALM, // 13 RING_FINGER_MCP
  "#70FA3B", // 14 RING_FINGER_PIP
  "#70FA3B", // 15 RING_FINGER_DIP
  "#70FA3B", // 16 RING_FINGER_TIP
  PALM, // 17 PINKY_MCP
  "#2D64BE", // 18 PINKY_PIP
  "#2D64BE", // 19 PINKY_DIP
  "#2D64BE", // 20 PINKY_TIP
];

const videoElement = document.getElementById("webcam");
const overlayCanvas = document.getElementById("overlay-canvas");
const overlayContext = overlayCanvas.getContext("2d");
const staticBar = document.getElementById("static-bar");
const staticLabel = document.getElementById("static-label");
const staticSecondary = document.getElementById("static-secondary");
const staticWarning = document.getElementById("static-warning");
const recordingBar = document.getElementById("recording-bar");
const recordingInfo = document.getElementById("recording-info");
const dynamicBar = document.getElementById("dynamic-bar");
const dynamicLabel = document.getElementById("dynamic-label");
const dynamicSecondary = document.getElementById("dynamic-secondary");
const recordButton = document.getElementById("record-button");
const statusMessage = document.getElementById("status-message");

let handLandmarker = null;
let websocket = null;
let awaitingServerResponse = false;
let latestOverlay = null;
let lastVideoTimestamp = -1;
let dynamicModelAvailable = false;
let isRunning = false;
// Whether a hand is visible in the most recently processed local frame.
// The live bars (static + RECORDING) are gated on this so they disappear the
// instant the hand leaves, matching the desktop app, instead of showing the
// last (stale) server overlay while sends are response-gated.
let handVisible = false;

/**
 * Update the status line shown to the user.
 * @param {string} message Text to display.
 */
function setStatus(message) {
  statusMessage.textContent = message;
}

/**
 * Strip the "letra_" prefix from a label for display, matching the
 * desktop overlay's "Letter: x" formatting.
 * @param {string} label Raw model label (e.g. "letra_a").
 * @returns {string} Display letter (e.g. "a").
 */
function displayLetter(label) {
  return label.startsWith("letra_") ? label.slice("letra_".length) : label;
}

/**
 * Initialise the MediaPipe HandLandmarker in VIDEO mode.
 * Mirrors the options used by the desktop `build_hand_landmarker()`.
 */
async function initialiseHandLandmarker() {
  const visionFileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_PATH);
  handLandmarker = await HandLandmarker.createFromOptions(visionFileset, {
    baseOptions: { modelAssetPath: HAND_LANDMARKER_MODEL_PATH },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.7,
  });
}

/**
 * Open the WebSocket connection to the gesture server.
 * Uses wss:// automatically when the page is served over HTTPS.
 */
function openWebSocket() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  websocket = new WebSocket(`${protocol}//${location.host}/ws`);

  websocket.addEventListener("open", () => {
    setStatus("Connected. Show a sign to the camera.");
  });

  websocket.addEventListener("message", (event) => {
    latestOverlay = JSON.parse(event.data);
    awaitingServerResponse = false;
  });

  websocket.addEventListener("close", () => {
    awaitingServerResponse = false;
    setStatus("Connection closed. Reload the page to reconnect.");
    recordButton.disabled = true;
  });

  websocket.addEventListener("error", () => {
    setStatus("Connection error. Reload the page to retry.");
  });
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
 * Send the current frame's landmarks to the server, if not already waiting.
 * Gating on the previous response keeps the socket from flooding and
 * decouples the (fast) draw rate from the (slower) round-trip rate.
 * @param {object|null} detection MediaPipe detection result for this frame.
 */
function sendToServer(detection) {
  if (
    !websocket ||
    websocket.readyState !== WebSocket.OPEN ||
    awaitingServerResponse
  ) {
    return;
  }

  const hasHand =
    detection && detection.worldLandmarks && detection.worldLandmarks.length > 0;

  const payload = {
    type: "frame",
    landmarks: hasHand ? toCoordinateArray(detection.worldLandmarks[0]) : null,
    handedness:
      hasHand && detection.handedness && detection.handedness.length > 0
        ? detection.handedness[0][0].categoryName
        : null,
    timestamp_ms: Math.round(performance.now()),
  };

  awaitingServerResponse = true;
  websocket.send(JSON.stringify(payload));
}

/**
 * Report whether a MediaPipe detection contains a visible hand this frame.
 * Uses the same image-space `landmarks` array that drives the skeleton so the
 * bars and the drawn hand agree on presence.
 * @param {object|null} detection MediaPipe detection result for this frame.
 * @returns {boolean} True if at least one hand was detected.
 */
function hasHand(detection) {
  return Boolean(
    detection && detection.landmarks && detection.landmarks.length > 0
  );
}

/**
 * Draw one hand's skeleton (connections + landmark dots) on the overlay
 * canvas, or clear it when no hand is visible.  Colors mirror
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
 * Update the static prediction bar (prefix "S"), including the
 * handedness/confidence line and the low-confidence warning.
 * @param {object} overlay Server overlay for the latest processed frame.
 */
function updateStaticBar(overlay) {
  if (!overlay.static_label) {
    staticBar.hidden = true;
    return;
  }
  staticBar.hidden = false;
  staticLabel.textContent = `S Letter: ${displayLetter(overlay.static_label)}`;

  const confidencePercent = (overlay.static_confidence * 100).toFixed(0);
  staticSecondary.textContent = overlay.handedness
    ? `Hand: ${overlay.handedness} | Confidence: ${confidencePercent}%`
    : `${confidencePercent}%`;
  staticWarning.hidden = overlay.static_confidence >= LOW_CONFIDENCE_THRESHOLD;
}

/**
 * Update the RECORDING indicator bar with mode, remaining time, and
 * buffered frame count, matching the desktop on-screen text.
 * @param {object} overlay Server overlay for the latest processed frame.
 */
function updateRecordingBar(overlay) {
  if (overlay.capture_state !== "RECORDING") {
    recordingBar.hidden = true;
    return;
  }
  recordingBar.hidden = false;
  recordingInfo.textContent = overlay.manual_capture
    ? `manual  |  ${overlay.buffer_length} frames`
    : `auto  |  ${overlay.recording_remaining_seconds.toFixed(1)}s remaining` +
      `  |  ${overlay.buffer_length} frames`;
}

/**
 * Update the latched dynamic prediction bar (prefix "D"). The server
 * handles the 3-second display window, so the bar simply mirrors the
 * overlay.
 * @param {object} overlay Server overlay for the latest processed frame.
 */
function updateDynamicBar(overlay) {
  if (!overlay.dynamic_label) {
    dynamicBar.hidden = true;
    return;
  }
  dynamicBar.hidden = false;
  dynamicLabel.textContent = `D Letter: ${displayLetter(overlay.dynamic_label)}`;
  dynamicSecondary.textContent =
    `${(overlay.dynamic_confidence * 100).toFixed(0)}%`;
}

/**
 * Keep the manual record button label in sync with the recording state.
 * @param {object} overlay Server overlay for the latest processed frame.
 */
function updateRecordButton(overlay) {
  recordButton.textContent =
    overlay.capture_state === "RECORDING"
      ? "Stop recording & classify"
      : "Start manual recording";
}

/**
 * The render loop: detect locally, draw the skeleton, stream landmarks,
 * and refresh the prediction bars once per animation frame.
 */
function renderLoop() {
  if (!isRunning) {
    return;
  }

  if (videoElement.readyState >= 2) {
    // Keep the canvas pixel size matched to the camera resolution.
    if (overlayCanvas.width !== videoElement.videoWidth) {
      overlayCanvas.width = videoElement.videoWidth;
      overlayCanvas.height = videoElement.videoHeight;
    }

    let detection = null;
    const timestamp = performance.now();
    // MediaPipe requires strictly increasing timestamps.
    if (timestamp > lastVideoTimestamp) {
      lastVideoTimestamp = timestamp;
      detection = handLandmarker.detectForVideo(videoElement, timestamp);
      // Refresh hand presence only on a fresh detection; on skipped
      // same-millisecond frames keep the previous value.
      handVisible = hasHand(detection);
    }

    drawSkeleton(detection);
    if (detection) {
      sendToServer(detection);
    }
    if (latestOverlay) {
      // The static and RECORDING bars are current-frame live state: gate them
      // on the local hand presence so they vanish the instant the hand leaves,
      // rather than lingering on the last (stale) server overlay.
      if (handVisible) {
        updateStaticBar(latestOverlay);
        updateRecordingBar(latestOverlay);
        updateRecordButton(latestOverlay);
      } else {
        staticBar.hidden = true;
        recordingBar.hidden = true;
        updateRecordButton({ capture_state: "IDLE" });
      }
      // The dynamic result is a server-latched, time-limited display window, so
      // it keeps showing briefly after the gesture (even with no hand) and then
      // disappears on its own once the server clears dynamic_label.
      updateDynamicBar(latestOverlay);
    }
  }

  window.requestAnimationFrame(renderLoop);
}

/**
 * Bootstrap: guard the secure context, load the MediaPipe model, query
 * model availability, start the camera, and enter the render loop.
 */
async function bootstrap() {
  // Browsers only expose the camera API on secure origins (HTTPS or
  // localhost). On e.g. http://0.0.0.0:8000 or a LAN IP,
  // navigator.mediaDevices is undefined — explain the fix instead of
  // showing a generic "denied" message.
  if (!window.isSecureContext || !navigator.mediaDevices) {
    setStatus(
      "Camera blocked: this page must be opened via http://localhost:" +
        `${location.port || "80"} (or HTTPS), not ${location.hostname}.`
    );
    return;
  }

  setStatus("Loading hand-landmark model…");
  try {
    await initialiseHandLandmarker();
  } catch (error) {
    setStatus("Could not load the MediaPipe hand-landmark model.");
    return;
  }

  try {
    const response = await fetch("/api/status");
    const modelStatus = await response.json();
    dynamicModelAvailable = modelStatus.dynamic_model_available;
    if (!modelStatus.static_model_available && !dynamicModelAvailable) {
      setStatus("Warning: no trained models are loaded on the server.");
    }
  } catch (error) {
    dynamicModelAvailable = false;
  }

  let mediaStream = null;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 } },
      audio: false,
    });
  } catch (error) {
    setStatus(
      "Camera access denied or unavailable. Allow camera access and reload."
    );
    return;
  }

  videoElement.srcObject = mediaStream;
  await new Promise((resolve) => {
    videoElement.onloadedmetadata = resolve;
  });

  openWebSocket();
  recordButton.disabled = !dynamicModelAvailable;

  isRunning = true;
  window.requestAnimationFrame(renderLoop);
}

recordButton.addEventListener("click", () => {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify({ type: "toggle_manual" }));
  }
});

bootstrap();
