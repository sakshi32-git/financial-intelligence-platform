"""
eye_tracker.py
==============
Webcam-based Eye Control Module for the Financial Intelligence Platform.

Uses the MediaPipe Tasks API (v0.10+) with FaceLandmarker for iris tracking
since the legacy mp.solutions API was removed in mediapipe >= 0.10.

Features:
- FaceLandmarker iris tracking for smooth, precise gaze-to-mouse mapping.
- Left-eye Eye Aspect Ratio (EAR) blink detection for left mouse click.
- Exponential Moving Average (EMA) cursor smoothing to prevent jitter.
- Blink cooldown timer to prevent accidental/repeated clicks.
- Multi-backend / multi-index webcam discovery (DSHOW, MSMF, ANY).
- Graceful camera permission / unavailable error handling.
"""

from __future__ import annotations

import math
import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

# Path to the downloaded MediaPipe FaceLandmarker model file
_MODEL_PATH = str(Path(__file__).resolve().parent / "face_landmarker.task")

# Lazy imports for cv2, mediapipe, pyautogui to prevent slow startup
_CV2 = None
_MP = None
_PYAUTOGUI = None


def _import_dependencies():
    """Lazily import OpenCV, MediaPipe, and PyAutoGUI."""
    global _CV2, _MP, _PYAUTOGUI
    if _CV2 is None:
        import cv2
        _CV2 = cv2
    if _MP is None:
        import mediapipe as mp
        _MP = mp
    if _PYAUTOGUI is None:
        import pyautogui
        _PYAUTOGUI = pyautogui
        _PYAUTOGUI.FAILSAFE = False
        _PYAUTOGUI.PAUSE = 0.0


class EyeTrackerController:
    """
    Controller for Eye Control Mode.
    Runs asynchronously in a background thread to map eye movements to the mouse cursor.
    Uses the MediaPipe Tasks API (FaceLandmarker) for iris/face landmark detection.
    """

    def __init__(self):
        self.is_running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Calibration & Smoothing Parameters
        self.smoothing_alpha: float = 0.22      # EMA smoothing weight (0.1 to 0.4)
        self.blink_threshold: float = 0.20      # Left eye EAR threshold for blink
        self.click_cooldown: float = 0.8        # Seconds between allowed clicks
        self.min_blink_duration: float = 0.08   # Minimum blink duration to register click

        # Sensitivity / Gaze Bounding Box (Normalized coords)
        self.gaze_min_x: float = 0.35
        self.gaze_max_x: float = 0.65
        self.gaze_min_y: float = 0.35
        self.gaze_max_y: float = 0.65

        # Runtime State
        self.status_message: str = "Normal Mouse Mode active"
        self.last_error: Optional[str] = None
        self.last_click_time: float = 0.0
        self.blink_start_time: Optional[float] = None
        self.current_cursor_pos: tuple = (0, 0)
        self.last_ear: float = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Return the current Eye Control status dict."""
        with self._lock:
            return {
                "active": self.is_running,
                "status_message": self.status_message,
                "error": self.last_error,
                "cursor_pos": self.current_cursor_pos,
                "last_ear": round(self.last_ear, 3),
            }

    def start(self) -> Dict[str, Any]:
        """Start Eye Control Mode in a background thread."""
        with self._lock:
            if self.is_running:
                return {"success": True, "message": "Eye Control Mode is already active."}

            try:
                _import_dependencies()
            except ImportError as exc:
                self.last_error = f"Missing required dependencies: {exc}"
                self.status_message = "Failed to load eye tracking dependencies."
                log.error(self.status_message, exc_info=True)
                return {"success": False, "message": self.last_error}

            if not os.path.exists(_MODEL_PATH):
                self.last_error = f"MediaPipe model file not found at: {_MODEL_PATH}"
                self.status_message = "Error: Model file missing"
                return {"success": False, "message": self.last_error}

            self.is_running = True
            self.last_error = None
            self.status_message = "Initialising camera and eye tracking..."
            self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self._thread.start()
            return {"success": True, "message": "Eye Control Mode activated."}

    def stop(self) -> Dict[str, Any]:
        """Stop Eye Control Mode."""
        with self._lock:
            if not self.is_running:
                return {"success": True, "message": "Normal Mouse Mode is active."}

            self.is_running = False
            self.status_message = "Normal Mouse Mode active"
            return {"success": True, "message": "Eye Control Mode deactivated."}

    def toggle(self) -> Dict[str, Any]:
        """Toggle between Normal Mouse Mode and Eye Control Mode."""
        if self.is_running:
            return self.stop()
        else:
            return self.start()

    def _compute_ear(self, landmarks, frame_w: int, frame_h: int) -> float:
        """
        Compute Left Eye Aspect Ratio (EAR) using FaceLandmarker landmarks.
        Left eye landmarks: 386/374 (vertical), 385/373 (vertical), 362/263 (horizontal).
        """
        def _pt(idx):
            lm = landmarks[idx]
            return (lm.x * frame_w, lm.y * frame_h)

        p386, p374 = _pt(386), _pt(374)
        p385, p373 = _pt(385), _pt(373)
        p362, p263 = _pt(362), _pt(263)

        d_v1 = math.hypot(p386[0] - p374[0], p386[1] - p374[1])
        d_v2 = math.hypot(p385[0] - p373[0], p385[1] - p373[1])
        d_h  = math.hypot(p362[0] - p263[0], p362[1] - p263[1])

        if d_h < 1e-6:
            return 0.3
        return (d_v1 + d_v2) / (2.0 * d_h)

    def _tracking_loop(self):
        """Main webcam eye tracking loop running in a background thread."""
        cv2 = _CV2
        mp = _MP
        pyautogui = _PYAUTOGUI

        screen_w, screen_h = pyautogui.size()
        prev_x, prev_y = screen_w // 2, screen_h // 2

        # ── Initialize Camera — multi-backend & multi-index fallback ───────────
        cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            for idx in (0, 1, 2):
                try:
                    c = cv2.VideoCapture(idx, backend)
                    if c and c.isOpened():
                        ret, test_frame = c.read()
                        if ret and test_frame is not None:
                            cap = c
                            log.info("Opened webcam index %d with backend %s", idx, backend)
                            break
                        else:
                            c.release()
                except Exception:
                    pass
            if cap is not None:
                break

        if not cap or not cap.isOpened():
            with self._lock:
                self.is_running = False
                self.last_error = "Webcam unavailable or permission denied"
                self.status_message = "Error: Webcam unavailable or permission denied"
            log.warning(self.last_error)
            return

        # ── Initialize MediaPipe FaceLandmarker (Tasks API) ───────────────────
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            with self._lock:
                self.is_running = False
                self.last_error = f"Failed to load FaceLandmarker: {exc}"
                self.status_message = f"Error: {self.last_error}"
            log.error("FaceLandmarker init error", exc_info=True)
            cap.release()
            return

        with self._lock:
            self.status_message = "Eye Control Mode active — tracking gaze..."

        try:
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                # Flip horizontally for natural mirror interaction
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run MediaPipe FaceLandmarker
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = face_landmarker.detect(mp_image)

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    # 1. Iris Gaze Position — Landmark 468 = Left Iris Center
                    iris = landmarks[468]
                    iris_x, iris_y = iris.x, iris.y

                    # Normalize gaze position relative to calibration bounds
                    norm_x = (iris_x - self.gaze_min_x) / (self.gaze_max_x - self.gaze_min_x)
                    norm_y = (iris_y - self.gaze_min_y) / (self.gaze_max_y - self.gaze_min_y)
                    norm_x = max(0.0, min(1.0, norm_x))
                    norm_y = max(0.0, min(1.0, norm_y))

                    target_x = norm_x * screen_w
                    target_y = norm_y * screen_h

                    # EMA Cursor Smoothing
                    curr_x = self.smoothing_alpha * target_x + (1.0 - self.smoothing_alpha) * prev_x
                    curr_y = self.smoothing_alpha * target_y + (1.0 - self.smoothing_alpha) * prev_y
                    prev_x, prev_y = curr_x, curr_y

                    target_int_x = int(curr_x)
                    target_int_y = int(curr_y)
                    pyautogui.moveTo(target_int_x, target_int_y)

                    with self._lock:
                        self.current_cursor_pos = (target_int_x, target_int_y)

                    # 2. Blink Detection for Left Click (via EAR)
                    ear = self._compute_ear(landmarks, w, h)
                    with self._lock:
                        self.last_ear = ear

                    now = time.time()
                    if ear < self.blink_threshold:
                        if self.blink_start_time is None:
                            self.blink_start_time = now
                    else:
                        if self.blink_start_time is not None:
                            blink_duration = now - self.blink_start_time
                            self.blink_start_time = None

                            if (blink_duration >= self.min_blink_duration and
                                    (now - self.last_click_time) >= self.click_cooldown):
                                pyautogui.click()
                                self.last_click_time = now
                                with self._lock:
                                    self.status_message = f"Click at ({target_int_x}, {target_int_y})"
                                log.info("Eye click at (%d, %d)", target_int_x, target_int_y)

                time.sleep(0.015)  # ~60 FPS

        except Exception as exc:
            with self._lock:
                self.last_error = f"Eye tracking error: {exc}"
                self.status_message = f"Error: {exc}"
            log.error("Eye tracking loop error: %s", exc, exc_info=True)
        finally:
            cap.release()
            face_landmarker.close()
            with self._lock:
                self.is_running = False
                if not self.last_error:
                    self.status_message = "Normal Mouse Mode active"


# Global singleton controller instance
eye_tracker = EyeTrackerController()
