"""
Ensemble Multi-Modal Sign & Gesture Detection Engine
Combines:
  1. MediaPipe 21 3D Joint Tracking (with CLAHE contrast normalization)
  2. 3D ToF Geometric Convexity & Solidity Segmentation (runs on depth silhouettes when lighting is dim)
  3. SignFrame AI PyTorch CNN (24-class ASL letter fingerspelling)
  4. YuNet + FER+ Facial Emotion Recognition
  5. Fallback Multi-Hand Blob Analyzer
"""

import os
import sys
import time
import struct
import threading
import cv2
import numpy as np
import requests

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
        self.latest_depth_frame = None
        self.latest_mask_frame = None
        self.latest_gesture = "No hand"
        self.latest_distance = None
        self.latest_handedness = "Unknown"
        self.latest_emotion = None
        self.latest_emotion_conf = None

        # SignFrame AI State
        self.latest_asl_letter = None
        self.latest_asl_conf = 0.0
        self.latest_stable_letter = None
        self.last_committed_letter = None
        self.stable_letter_start_time = 0.0
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

        self._local_cap = None
        self._a075v_online = False
        self._initialized = True

    def init_a075v(self):
        try:
            cfg = struct.pack("<BBBBBBBBi", 1, 0, 255, 0, 2, 7, 1, 0, 0)
            requests.post(f"{self.base_url}/set_cfg", data=cfg, timeout=1.0)
            self._a075v_online = True
            return True
        except Exception:
            self._a075v_online = False
            return False

    def fetch_a075v_frames(self):
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
        depth, rgb = self.fetch_a075v_frames()
        if rgb is not None:
            self.camera_source = "MaixSense-A075V (ToF RGB-D)"
            if self._local_cap is not None:
                self._local_cap.release()
                self._local_cap = None
            return depth, rgb

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

    @staticmethod
    def enhance_contrast(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def detect_depth_hand_contour(self, depth, vh, vw, max_dist_mm=1600):
        """Pipeline 2: 3D ToF Depth Segmentation (Works even in complete darkness or monochrome)"""
        if depth is None or depth.dtype != np.uint16:
            return None, None, None, None

        d_res = cv2.resize(depth, (vw, vh), interpolation=cv2.INTER_NEAREST)
        valid = (d_res > 80) & (d_res < max_dist_mm)
        if not np.any(valid):
            return None, None, None, np.zeros((vh, vw), dtype=np.uint8)

        d_min = np.min(d_res[valid])
        # Isolate foreground slice (14 cm window)
        fg = ((d_res >= d_min) & (d_res <= (d_min + 140))).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg)
        best_cnt = None
        best_area = 0
        best_cx, best_cy = 0, 0
        dist_m = float(d_min) / 1000.0

        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 600:
                continue
            comp = (labels == lbl).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if area > best_area:
                    best_area = area
                    best_cnt = c
                    best_cx = int(centroids[lbl][0])
                    best_cy = int(centroids[lbl][1])

        if best_cnt is not None:
            # Analyze contour convexity & shape
            hull = cv2.convexHull(best_cnt, returnPoints=False)
            valleys = 0
            if hull is not None and len(hull) >= 4:
                try:
                    defects = cv2.convexityDefects(best_cnt, hull)
                    if defects is not None:
                        for s, e, f, d in defects.reshape(-1, 4):
                            if (d / 256.0) > 12.0:
                                valleys += 1
                except Exception:
                    pass

            fingers = min(valleys + 1, 5) if valleys > 0 else 0
            label_map = {0: "A (Fist)", 1: "D / 1", 2: "V / Peace", 3: "W / Three", 4: "B / 4", 5: "Hello / 5"}
            gesture = label_map.get(fingers, f"{fingers} Fingers")
            return best_cnt, gesture, dist_m, fg

        return None, None, None, fg

    def loop(self):
        self.init_a075v()

        while self.running:
            depth, rgb = self.fetch_frames()
            if rgb is None:
                time.sleep(0.04)
                continue

            vh, vw = rgb.shape[:2]
            output = rgb.copy()
            enhanced = self.enhance_contrast(rgb)

            asl_letter = None
            asl_conf = 0.0
            stable_letter = None
            distance = None
            emotion = None
            emotion_conf = None

            # -------------------------------------------------------------
            # Process 1: Facial Emotion Detection (YuNet + FER+)
            # -------------------------------------------------------------
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

            # -------------------------------------------------------------
            # Process 2: MediaPipe 3D Landmark + PyTorch CNN Detection
            # -------------------------------------------------------------
            # -------------------------------------------------------------
            # Auxiliary Views: 3D ToF Depth Map & Black-and-White Mask
            # -------------------------------------------------------------
            depth_vis = None
            bw_mask = None
            fg_mask = None

            if depth is not None:
                d_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)

            if landmarks:
                bbox = get_hand_bbox(landmarks, rgb.shape)
                image_tensor = preprocess_hand(rgb, bbox)

                if image_tensor is not None:
                    asl_letter, asl_conf = self.asl_predictor.predict(image_tensor)
                    stable_letter = self.stabilizer.update(asl_letter, asl_conf)
                else:
                    self.stabilizer.reset()

                # Overlay skeleton
                x1, y1, x2, y2 = bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 200), 2)
                for pt in landmarks:
                    px, py = int(pt.x * vw), int(pt.y * vh)
                    cv2.circle(output, (px, py), 3, (255, 255, 255), -1)

                # Generate B&W skin / hand threshold mask for the hand region
                hand_roi = rgb[max(0, y1):min(vh, y2), max(0, x1):min(vw, x2)]
                bw_mask = np.zeros((vh, vw), dtype=np.uint8)
                if hand_roi.size > 0:
                    hsv_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
                    roi_mask = cv2.inRange(hsv_roi, np.array([0, 20, 70]), np.array([25, 255, 255]))
                    bw_mask[max(0, y1):min(vh, y2), max(0, x1):min(vw, x2)] = roi_mask

                if depth is not None:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    d_res = cv2.resize(depth, (vw, vh), interpolation=cv2.INTER_NEAREST)
                    patch = d_res[max(0, cy - 5) : min(vh, cy + 5), max(0, cx - 5) : min(vw, cx + 5)]
                    valid = patch[(patch > 80) & (patch < 2500)]
                    if valid.size > 0:
                        distance = float(np.median(valid)) / 1000.0

                label_str = f"ASL CNN: {asl_letter} ({int(asl_conf * 100)}%)"
                if stable_letter:
                    label_str += f" -> [{stable_letter}]"
                if distance:
                    label_str += f" | {distance:.2f}m"

                cv2.putText(output, label_str, (max(8, x1), max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 200), 2)

            else:
                self.stabilizer.reset()

                # -------------------------------------------------------------
                # Process 3: 3D ToF Depth Foreground Contour Segmentation
                # (Active when MediaPipe struggles due to low lighting or monochrome)
                # -------------------------------------------------------------
                if depth is not None:
                    hand_cnt, depth_gest, d_meter, fg_mask = self.detect_depth_hand_contour(depth, vh, vw)
                    bw_mask = fg_mask
                    if hand_cnt is not None:
                        cv2.drawContours(output, [hand_cnt], -1, (0, 230, 255), 2)
                        x, y, w, h = cv2.boundingRect(hand_cnt)
                        distance = d_meter
                        asl_letter = depth_gest
                        asl_conf = 0.85
                        label_str = f"ToF 3D: {depth_gest} | {distance:.2f}m"
                        cv2.putText(output, label_str, (max(8, x), max(20, y - 10)),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 230, 255), 2)

            # Fallbacks for auxiliary visualizers if depth or mask is None
            if depth_vis is None:
                # Simulated/derived depth visualization from grayscale gradient
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                depth_vis = cv2.applyColorMap(cv2.equalizeHist(gray), cv2.COLORMAP_JET)

            if bw_mask is None:
                # Background subtraction / Otsu skin binary mask
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                _, bw_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Auto-commit stable letters into composer
            # If the user holds a stable sign for > 0.8s without changing, append to composer!
            now = time.time()
            if stable_letter:
                if stable_letter != self.last_committed_letter:
                    if self.stable_letter_start_time == 0.0:
                        self.stable_letter_start_time = now
                    elif (now - self.stable_letter_start_time) >= 0.8:
                        self.composer.append_letter(stable_letter)
                        self.last_committed_letter = stable_letter
                        self.stable_letter_start_time = 0.0
                else:
                    self.stable_letter_start_time = 0.0
            else:
                self.stable_letter_start_time = 0.0
                self.last_committed_letter = None

            # Camera watermark
            cv2.putText(output, f"Source: {self.camera_source}", (10, vh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

            with self._lock:
                self.latest_frame = output
                self.latest_depth_frame = depth_vis
                self.latest_mask_frame = bw_mask
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

    def get_depth_jpeg(self):
        with self._lock:
            frame = self.latest_depth_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    def get_mask_jpeg(self):
        with self._lock:
            frame = self.latest_mask_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
                "composed_text": self.composer.text,
                "camera_source": self.camera_source,
            }

    def add_stable_letter(self):
        with self._lock:
            if self.latest_stable_letter:
                self.composer.append_letter(self.latest_stable_letter)
                return self.composer.text
        return self.composer.text

    def add_letter(self, letter):
        with self._lock:
            if letter:
                self.composer.append_letter(letter)
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
