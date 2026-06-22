/*
 * app.js — browser front-end for the Nahual LSM gesture demo.
 *
 * Responsibilities (deliberately thin — no gesture logic lives here):
 *   1. Run MediaPipe HandLandmarker locally to obtain hand landmarks.
 *   2. Draw the camera feed and landmarks on a canvas.
 *   3. Stream the metric `worldLandmarks` over a WebSocket to the FastAPI
 *      server, which performs all feature extraction and classification.
 *   4. Render the static (S) and dynamic (D) predictions the server returns.
 */

import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/vision_bundle.mjs";

const MEDIAPIPE_WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm";
const HAND_LANDMARKER_MODEL_PATH = "/models/hand_landmarker.task";

// MediaPipe hand skeleton: pairs of landmark indices to connect when drawing.
// Hardcoded so we don't depend on DrawingUtils, which is not exported by the
// tasks-vision ESM bundle.
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [5, 9], [9, 10], [10, 11], [11, 12], // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17], // palm base
];

// Per-landmark colors, indexed by MediaPipe landmark id (0..20).
// These mirror LANDMARK_NAMES in nahual/visualization.py, converted from
// OpenCV BGR tuples to canvas RGB hex (palm/MCP joints are gray; each finger's
// distal joints share that finger's color).
const PALM = "#808080"; // WRIST, CMC, and all MCP joints
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

// DOM references.
const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("overlay");
const canvasContext = canvasElement.getContext("2d");
const toggleButton = document.getElementById("toggle-button");
const debugCheckbox = document.getElementById("debug-checkbox");
const statusElement = document.getElementById("status");

// Runtime state.
let handLandmarker = null;
let webSocket = null;
let mediaStream = null;
let isRunning = false;
let awaitingServerResponse = false;
let latestOverlay = null;
let lastVideoTimestamp = -1;

/**
 * Update the status line shown to the user.
 * @param {string} message Text to display.
 */
function setStatus(message) {
  statusElement.textContent = message;
}

/**
 * Format a model label such as "letra_a" into a display string like "A".
 * @param {string} label Raw model label.
 * @returns {string} Human-friendly label.
 */
function formatLabel(label) {
  return label.replace(/^letra_/, "").toUpperCase();
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
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  webSocket = new WebSocket(`${protocol}://${window.location.host}/ws`);

  webSocket.addEventListener("message", (event) => {
    latestOverlay = JSON.parse(event.data);
    awaitingServerResponse = false;
  });

  webSocket.addEventListener("close", () => {
    awaitingServerResponse = false;
  });
}

/**
 * Start the camera, the WebSocket, and the render loop.
 */
async function start() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (error) {
    setStatus("No se pudo acceder a la cámara. Revisa los permisos.");
    return;
  }

  videoElement.srcObject = mediaStream;
  await videoElement.play();

  openWebSocket();

  isRunning = true;
  toggleButton.textContent = "Detener";
  setStatus("Detectando… muestra una seña a la cámara.");
  window.requestAnimationFrame(renderLoop);
}

/**
 * Stop the camera, close the WebSocket, and clear the canvas.
 */
function stop() {
  isRunning = false;
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (webSocket) {
    webSocket.close();
    webSocket = null;
  }
  latestOverlay = null;
  canvasContext.clearRect(0, 0, canvasElement.width, canvasElement.height);
  toggleButton.textContent = "Iniciar cámara";
  setStatus("Detenido.");
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
    !webSocket ||
    webSocket.readyState !== WebSocket.OPEN ||
    awaitingServerResponse
  ) {
    return;
  }

  const hasHand =
    detection && detection.worldLandmarks && detection.worldLandmarks.length > 0;

  const payload = hasHand
    ? {
        landmarks: toCoordinateArray(detection.worldLandmarks[0]),
        handedness:
          detection.handedness && detection.handedness.length > 0
            ? detection.handedness[0][0].categoryName
            : null,
        timestamp_ms: Math.round(performance.now()),
      }
    : { landmarks: null, handedness: null, timestamp_ms: Math.round(performance.now()) };

  awaitingServerResponse = true;
  webSocket.send(JSON.stringify(payload));
}

/**
 * Draw the mirrored camera frame and the detected hand landmarks.
 * @param {object|null} detection MediaPipe detection result for this frame.
 */
function drawScene(detection) {
  const width = canvasElement.width;
  const height = canvasElement.height;

  canvasContext.save();
  // Mirror horizontally for a natural selfie view (display only — the
  // detector and server always see the unmirrored frame).
  canvasContext.translate(width, 0);
  canvasContext.scale(-1, 1);
  canvasContext.drawImage(videoElement, 0, 0, width, height);

  if (detection && detection.landmarks) {
    for (const landmarks of detection.landmarks) {
      drawHand(landmarks, width, height);
    }
  }
  canvasContext.restore();
}

/**
 * Draw one hand's skeleton (connections + landmark dots) on the canvas.
 * @param {Array<{x:number,y:number}>} landmarks Normalized image-space landmarks.
 * @param {number} width Canvas width in pixels.
 * @param {number} height Canvas height in pixels.
 */
function drawHand(landmarks, width, height) {
  // Connections.
  canvasContext.strokeStyle = "#4ade80";
  canvasContext.lineWidth = 3;
  for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    canvasContext.beginPath();
    canvasContext.moveTo(start.x * width, start.y * height);
    canvasContext.lineTo(end.x * width, end.y * height);
    canvasContext.stroke();
  }

  // Landmark dots, colored per finger to match LANDMARK_NAMES in
  // nahual/visualization.py.
  for (let index = 0; index < landmarks.length; index += 1) {
    const point = landmarks[index];
    canvasContext.fillStyle = LANDMARK_COLORS[index] || "#ffffff";
    canvasContext.beginPath();
    canvasContext.arc(point.x * width, point.y * height, 4, 0, Math.PI * 2);
    canvasContext.fill();
  }
}

/**
 * Draw a two-line prediction block (label + secondary info) and return the
 * total height consumed so the next block can stack beneath it.
 * @param {string} titleLine  Primary line, e.g. "Static: Letter a".
 * @param {string} detailLine Secondary line, e.g. "Hand: Left | Confidence: 89%".
 * @param {string} titleColor Color for the primary line.
 * @param {string} detailColor Color for the secondary line.
 * @param {number} x Left position in pixels.
 * @param {number} y Top position in pixels.
 * @returns {number} Total block height in pixels.
 */
function drawPredictionBlock(titleLine, detailLine, titleColor, detailColor, x, y) {
  const padding = 8;
  const titleFontSize = 24;
  const detailFontSize = 16;
  const lineGap = 6;

  canvasContext.font = `600 ${titleFontSize}px system-ui, sans-serif`;
  const titleMetrics = canvasContext.measureText(titleLine);
  canvasContext.font = `400 ${detailFontSize}px system-ui, sans-serif`;
  const detailMetrics = canvasContext.measureText(detailLine);

  const blockWidth = Math.max(titleMetrics.width, detailMetrics.width) + padding * 2;
  const blockHeight = titleFontSize + lineGap + detailFontSize + padding * 2;

  canvasContext.fillStyle = "rgba(30, 30, 30, 0.85)";
  canvasContext.fillRect(x, y, blockWidth, blockHeight);

  canvasContext.textBaseline = "top";
  canvasContext.font = `600 ${titleFontSize}px system-ui, sans-serif`;
  canvasContext.fillStyle = titleColor;
  canvasContext.fillText(titleLine, x + padding, y + padding);

  canvasContext.font = `400 ${detailFontSize}px system-ui, sans-serif`;
  canvasContext.fillStyle = detailColor;
  canvasContext.fillText(detailLine, x + padding, y + padding + titleFontSize + lineGap);

  return blockHeight + 6;
}

/**
 * Draw the latest static / dynamic predictions and optional motion debug.
 */
function drawPredictions() {
  if (!latestOverlay) {
    return;
  }

  let blockY = 12;

  if (latestOverlay.static_label) {
    const title = `Static: Letter ${formatLabel(latestOverlay.static_label)}`;
    const confidencePercent = `${Math.round(latestOverlay.static_confidence * 100)}%`;
    const detail = latestOverlay.handedness
      ? `Hand: ${latestOverlay.handedness} | Confidence: ${confidencePercent}`
      : `Confidence: ${confidencePercent}`;
    blockY += drawPredictionBlock(title, detail, "#ffffff", "#dcdcdc", 12, blockY);
  }

  if (latestOverlay.dynamic_label) {
    const title = `Dynamic: Letter ${formatLabel(latestOverlay.dynamic_label)}`;
    const confidencePercent = `${Math.round(latestOverlay.dynamic_confidence * 100)}%`;
    const detail = `Confidence: ${confidencePercent}`;
    blockY += drawPredictionBlock(title, detail, "#ffffff", "#dcdcdc", 12, blockY);
  }

  if (debugCheckbox.checked) {
    drawMotionDebug();
  }
}

/**
 * Draw a line of text with a translucent dark background for readability.
 * @param {string} text Text to draw.
 * @param {number} x Left position in pixels.
 * @param {number} y Top position in pixels.
 * @param {string} color Text colour.
 */
function drawTextWithBackground(text, x, y, color) {
  const metrics = canvasContext.measureText(text);
  const padding = 6;
  canvasContext.fillStyle = "rgba(0, 0, 0, 0.55)";
  canvasContext.fillRect(x - padding, y - padding, metrics.width + padding * 2, 30);
  canvasContext.fillStyle = color;
  canvasContext.fillText(text, x, y);
}

/**
 * Draw the motion-debug readout in the bottom-left corner.
 * Mirrors the desktop `draw_motion_debug` output.
 */
function drawMotionDebug() {
  const lines = [
    `raw motion : ${latestOverlay.raw_motion.toFixed(4)}`,
    `smoothed   : ${latestOverlay.smoothed_motion.toFixed(4)}`,
    `state      : ${latestOverlay.capture_state}  buf=${latestOverlay.buffer_length}`,
  ];
  canvasContext.font = "14px monospace";
  let y = canvasElement.height - lines.length * 18 - 8;
  for (const line of lines) {
    drawTextWithBackground(line, 8, y, "#4ade80");
    y += 18;
  }
}

/**
 * The render loop: detect, draw, and stream landmarks once per animation frame.
 */
function renderLoop() {
  if (!isRunning) {
    return;
  }

  if (videoElement.readyState >= 2) {
    // Keep the canvas pixel size matched to the camera resolution.
    if (canvasElement.width !== videoElement.videoWidth) {
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
    }

    let detection = null;
    const timestamp = performance.now();
    // MediaPipe requires strictly increasing timestamps.
    if (timestamp > lastVideoTimestamp) {
      lastVideoTimestamp = timestamp;
      detection = handLandmarker.detectForVideo(videoElement, timestamp);
    }

    drawScene(detection);
    drawPredictions();
    if (detection) {
      sendToServer(detection);
    }
  }

  window.requestAnimationFrame(renderLoop);
}

/**
 * Toggle the demo on/off from the button.
 */
function onToggle() {
  if (isRunning) {
    stop();
  } else {
    start();
  }
}

/**
 * Bootstrap: load the model, query model availability, wire up controls.
 */
async function bootstrap() {
  toggleButton.disabled = true;
  try {
    await initialiseHandLandmarker();
  } catch (error) {
    setStatus("No se pudo cargar el modelo de MediaPipe.");
    return;
  }

  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    if (!status.static_model_available && !status.dynamic_model_available) {
      setStatus("Advertencia: no hay modelos entrenados cargados en el servidor.");
    } else {
      const parts = [];
      if (status.static_model_available) parts.push("estático");
      if (status.dynamic_model_available) parts.push("dinámico");
      setStatus(`Listo. Modelos cargados: ${parts.join(" + ")}. Pulsa Iniciar.`);
    }
  } catch (error) {
    setStatus("Listo. Pulsa Iniciar cámara.");
  }

  toggleButton.disabled = false;
  toggleButton.addEventListener("click", onToggle);
}

bootstrap();
