"""
nahual

LSM (Lengua de Señas Mexicana) gesture detection package.

Modules:
    sign_language      -- Supported sign languages and the -lsm / -asl flag.
    hand_landmarker    -- Canonical MediaPipe HandLandmarker configuration.
    gesture_heuristics -- Landmark preprocessing and feature extraction.
    gesture_trainer    -- Model training and inference.
    realtime_session   -- Stateful, frame-by-frame recognition session.
    gesture_collector  -- Interactive data collection tool.
    visualization      -- OpenCV drawing helpers (landmarks, overlays).
    data_inspector     -- Dataset inspection utilities (sample counts per label).
"""
