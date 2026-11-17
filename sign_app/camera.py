"""
Enhanced Camera Processor:
- Dual Camera Support: Sipeed MaixSense-A075V (ToF RGB-D) with seamless fallback to Local Webcam (index 0).
- SignFrame AI Integration: Real-time 24-letter ASL fingerspelling recognition via PyTorch CNN.
- 3D Depth Isolation: Background noise removal when ToF camera is connected.
- Facial Emotion Detection: YuNet + FER+ ONNX runtime.
- Background Thread Loop: Singleton design to prevent OOM/re-instantiation.
"""

import os
import sys
import time
import struct
import threading
import cv2
import numpy as np
import requests

# Set search paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sign_app.emotion_detector import EmotionDetector
from src.hand_tracker import HandTracker
from src.inference import SignPredictor
from src.preprocessing import get_hand_bbox, preprocess_hand
from src.stabilizer import PredictionStabilizer
from src.text_composer import TextComposer
from src.communication import SpeechEngine


class CameraProcessor:
    _instance = None
    _lock_init = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Thread-safe Singleton so FastAPI endpoints share a single camera pipeline."""
        with cls._lock_init:
            if cls._instance is None:
                cls._instance = super(CameraProcessor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, rgb_url="http://192.168.233.1/rgb", depth_url="http://192.168.233.1/depth"):
        if getattr(self, "_initialized", False):
            return

        self.rgb_url = rgb_url
        self.depth_url = depth_url
        if "://" in rgb_url:
            self.base_url = rgb_url.split("/")[0] + "//" + rgb_url.split("/")[2]
        else:
            self.base_url = "http://192.168.233.1"

        self._lock = threading.Lock()
        self.latest_frame = None
        self.latest_gesture = "No hand"
        self.latest_distance = None
        self.latest_handedness = "Unknown"
        self.latest_emotion = None
        self.latest_emotion_conf = None

        # SignFrame AI State
        self.latest_asl_letter = None
        self.latest_asl_conf = 0.0
        self.latest_stable_letter = None
        self.composed_text = ""
        self.camera_source = "Detecting..."

        self.running = False
        self.thread = None

        # Detectors
        self.emotion_detector = EmotionDetector()
        self.hand_tracker = HandTracker()
        self.asl_predictor = SignPredictor()
        self.stabilizer = PredictionStabilizer()
        self.composer = TextComposer()
        self.speech = SpeechEngine()

        # Local webcam fallback
        self._local_cap = None
        self._a075v_online = False
        self._initialized = True

    def init_a075v(self):
        """Send initialization packet to MaixSense-A075V if reachable."""
        try:
            cfg = struct.pack("<BBBBBBBBi", 1, 0, 255, 0, 2, 7, 1, 0, 0)
            requests.post(f"{self.base_url}/set_cfg", data=cfg, timeout=1.0)
            self._a075v_online = True
            return True
        except Exception:
            self._a075v_online = False
            return False

    def fetch_a075v_frames(self):
        """Fetch depth and RGB from A075V getdeep endpoint."""
        try:
            resp = requests.get(f"{self.base_url}/getdeep", timeout=1.2)
            data = resp.content
            if len(data) >= 36:
                _, deep_mode, _, ir_mode, status_mode, _, rgb_mode, _, _ = struct.unpack("<BBBBBBBBi", data[16:28])
                _, rgb_data_size = struct.unpack("<ii", data[28:36])
                payload = data[36:]

                deepth_size = (320 * 240 * 2) >> deep_mode
                ir_size = (320 * 240 * 2) >> ir_mode
                status_size = (320 * 240 // 8) * (
                    16 if status_mode == 0 else 2 if status_mode == 1 else 8 if status_mode == 2 else 1
                )

                depth = None
                if deepth_size > 0 and len(payload) >= deepth_size:
                    depth_bytes = payload[:deepth_size]
                    dtype = np.uint16 if deep_mode == 0 else np.uint8
                    depth = np.frombuffer(depth_bytes, dtype=dtype).reshape((240, 320))

                rgb_offset = deepth_size + ir_size + status_size
                rgb_bytes = payload[rgb_offset : rgb_offset + rgb_data_size]
                rgb = None
                if rgb_data_size > 0 and len(rgb_bytes) == rgb_data_size:
                    rgb = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                if rgb is not None:
                    self._a075v_online = True
                    return depth, rgb
        except Exception:
            self._a075v_online = False
        return None, None

    def fetch_frames(self):
        """Try A075V ToF camera first; smoothly fallback to local USB/integrated webcam."""
        depth, rgb = self.fetch_a075v_frames()
        if rgb is not None:
            self.camera_source = "MaixSense-A075V (ToF RGB-D)"
            return depth, rgb

        # Fallback to local webcam
        self.camera_source = "Webcam (Local Video0)"
        if self._local_cap is None or not self._local_cap.isOpened():
            self._local_cap = cv2.VideoCapture(0)
            self._local_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._local_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if self._local_cap.isOpened():
            ret, frame = self._local_cap.read()
            if ret and frame is not None:
                return None, cv2.flip(frame, 1)

        return None, None

    def isolate_depth_foreground(self, rgb, depth, depth_window_mm=130):
        """Slice foreground hand silhouette using 3D millimeter depth map."""
        if depth is None:
            return rgb

        vh, vw = rgb.shape[:2]
        d_res = cv2.resize(depth, (vw, vh), interpolation=cv2.INTER_NEAREST)
        valid = (d_res > 100) & (d_res < 2000)
        if not np.any(valid):
            return rgb

        d_min = np.min(d_res[valid])
        fg_mask = (d_res >= d_min) & (d_res <= (d_min + depth_window_mm))
        fg_mask_u8 = (fg_mask * 255).astype(np.uint8)

        # Apply mask to clean background clutter
        isolated = cv2.bitwise_and(rgb, rgb, mask=fg_mask_u8)
        return isolated

    def loop(self):
        self.init_a075v()

        while self.running:
            depth, rgb = self.fetch_frames()
            if rgb is None:
                time.sleep(0.04)
                continue

            vh, vw = rgb.shape[:2]
            output = rgb.copy()

            asl_letter = None
            asl_conf = 0.0
            stable_letter = None
            distance = None
            emotion = None
            emotion_conf = None

            # 1. Face Emotion Detection (YuNet + FER+)
            try:
                emo, conf, face_box = self.emotion_detector.detect(rgb)
                if emo is not None:
                    emotion = emo
                    emotion_conf = conf
                    if face_box:
                        x, y, fw, fh = face_box
                        cv2.rectangle(output, (x, y), (x + fw, y + fh), (255, 0, 255), 2)
                        cv2.putText(output, f"{emotion} ({int(conf * 100)}%)",
                                    (x, max(y - 8, 15)),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)
            except Exception:
                pass

            # 2. Hand Detection & SignFrame AI ASL Prediction
            clean_input = rgb
            if depth is not None:
                # Use ToF depth foreground isolation to eliminate background bleed
                clean_input = self.isolate_depth_foreground(rgb, depth)

            landmarks = self.hand_tracker.detect(clean_input)
            if landmarks:
                bbox = get_hand_bbox(landmarks, clean_input.shape)
                image_tensor = preprocess_hand(clean_input, bbox)

                if image_tensor is not None:
                    asl_letter, asl_conf = self.asl_predictor.predict(image_tensor)
                    stable_letter = self.stabilizer.update(asl_letter, asl_conf)
                else:
                    self.stabilizer.reset()

                # Visual overlay on hand
                x1, y1, x2, y2 = bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 200), 2)
                for pt in landmarks:
                    px, py = int(pt.x * vw), int(pt.y * vh)
                    cv2.circle(output, (px, py), 3, (255, 255, 255), -1)

                # Distance estimation if depth available
                if depth is not None:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    d_res = cv2.resize(depth, (vw, vh), interpolation=cv2.INTER_NEAREST)
                    patch = d_res[max(0, cy - 5) : min(vh, cy + 5), max(0, cx - 5) : min(vw, cx + 5)]
                    valid = patch[(patch > 80) & (patch < 2500)]
                    if valid.size > 0:
                        distance = float(np.median(valid)) / 1000.0

                label_str = f"ASL: {asl_letter} ({int(asl_conf * 100)}%)"
                if stable_letter:
                    label_str += f" -> [{stable_letter}]"
                if distance:
                    label_str += f" | {distance:.2f}m"

                cv2.putText(output, label_str, (max(8, x1), max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 200), 2)
            else:
                self.stabilizer.reset()

            # Status watermark
            cv2.putText(output, f"Source: {self.camera_source}", (10, vh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            with self._lock:
                self.latest_frame = output
                self.latest_gesture = stable_letter or asl_letter or "No hand"
                self.latest_asl_letter = asl_letter
                self.latest_asl_conf = asl_conf
                self.latest_stable_letter = stable_letter
                self.latest_distance = distance
                self.latest_emotion = emotion
                self.latest_emotion_conf = emotion_conf
                self.composed_text = self.composer.text

            time.sleep(0.02)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self._local_cap is not None:
            self._local_cap.release()
            self._local_cap = None
        if self.thread:
            self.thread.join(timeout=1.5)

    def get_frame_jpeg(self):
        with self._lock:
            frame = self.latest_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else None

    def get_data(self):
        with self._lock:
            return {
                "gesture": self.latest_gesture,
                "asl_letter": self.latest_asl_letter,
                "asl_conf": self.latest_asl_conf,
                "stable_letter": self.latest_stable_letter,
                "distance": self.latest_distance,
                "emotion": self.latest_emotion,
                "emotion_conf": self.latest_emotion_conf,
                "composed_text": self.composed_text,
                "camera_source": self.camera_source,
            }

    def add_stable_letter(self):
        with self._lock:
            if self.latest_stable_letter:
                self.composer.add_letter(self.latest_stable_letter)
                return self.composer.text
        return self.composer.text

    def add_space(self):
        with self._lock:
            self.composer.add_space()
            return self.composer.text

    def backspace(self):
        with self._lock:
            self.composer.backspace()
            return self.composer.text

    def clear_text(self):
        with self._lock:
            self.composer.clear()
            return self.composer.text

    def speak_text(self):
        with self._lock:
            text = self.composer.text
        return self.speech.speak(text)
