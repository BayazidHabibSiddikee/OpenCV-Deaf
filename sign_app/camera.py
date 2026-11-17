"""Camera processor: RGB+depth from MaixSense-A075V + OpenCV hand gesture + emotion detection."""
import cv2
import numpy as np
import requests
import struct
import threading
import time
import sys

sys.path.insert(0, "/home/bayazid/Documents/OpenCV/sign_app")
from emotion_detector import EmotionDetector


class CameraProcessor:
    def __init__(self, rgb_url="http://192.168.233.1/rgb", depth_url="http://192.168.233.1/depth"):
        self.rgb_url = rgb_url
        self.depth_url = depth_url
        # Extract base host from rgb_url or default
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

        self.running = False
        self.thread = None

        self.LOWER_SKIN = np.array([0, 20, 70], dtype=np.uint8)
        self.UPPER_SKIN = np.array([50, 255, 255], dtype=np.uint8)
        self.MIN_HAND_AREA = 3000

        self.DEPTH_SAMPLE_RADIUS = 4
        self.DEPTH_IS_RAW_UINT16 = True
        self.DEPTH_MAX_METERS = 2.0

        self.emotion_detector = EmotionDetector()
        self._cfg_initialized = False

    def init_camera(self):
        """Send configuration to A075V if needed."""
        try:
            # trigger_mode=1 (auto), deep_mode=0 (16-bit), deep_shift=255, ir_mode=0,
            # status_mode=2 (8-bit), status_mask=7, rgb_mode=1 (JPEG), rgb_res=0 (640x480), expose_time=0
            cfg = struct.pack("<BBBBBBBBi", 1, 0, 255, 0, 2, 7, 1, 0, 0)
            requests.post(f"{self.base_url}/set_cfg", data=cfg, timeout=3)
            self._cfg_initialized = True
        except Exception as e:
            pass

    def fetch_frames(self):
        """Fetch depth and RGB simultaneously from A075V getdeep endpoint or fallback to separate endpoints."""
        try:
            resp = requests.get(f"{self.base_url}/getdeep", timeout=3)
            data = resp.content
            if len(data) >= 36:
                trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time = struct.unpack('<BBBBBBBBi', data[16:28])
                deep_data_size, rgb_data_size = struct.unpack('<ii', data[28:36])
                payload = data[36:]

                deepth_size = (320 * 240 * 2) >> deep_mode
                ir_size = (320 * 240 * 2) >> ir_mode
                status_size = (320 * 240 // 8) * (16 if status_mode == 0 else 2 if status_mode == 1 else 8 if status_mode == 2 else 1)

                depth_bytes = payload[:deepth_size]
                dtype = np.uint16 if deep_mode == 0 else np.uint8
                depth = np.frombuffer(depth_bytes, dtype=dtype).reshape((240, 320)) if deepth_size > 0 else None
                self.DEPTH_IS_RAW_UINT16 = (deep_mode == 0)

                rgb_offset = deepth_size + ir_size + status_size
                rgb_bytes = payload[rgb_offset : rgb_offset + rgb_data_size]
                rgb = None
                if rgb_data_size > 0 and len(rgb_bytes) == rgb_data_size:
                    if rgb_mode == 1:
                        rgb = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                if rgb is not None:
                    return depth, rgb
        except Exception:
            pass

        # Fallback to separate endpoints if getdeep fails
        return self.fetch_depth(), self.fetch_rgb()

    def fetch_rgb(self):
        try:
            resp = requests.get(self.rgb_url, timeout=3)
            resp.raise_for_status()
            if len(resp.content) == 0:
                # If /rgb is empty, use getdeep
                _, rgb = self.fetch_frames()
                return rgb
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def fetch_depth(self):
        try:
            resp = requests.get(self.depth_url, timeout=3)
            resp.raise_for_status()
            if len(resp.content) == 0:
                depth, _ = self.fetch_frames()
                return depth
            if self.DEPTH_IS_RAW_UINT16:
                arr = np.frombuffer(resp.content, dtype=np.uint16)
                return arr.reshape((240, 320))
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            d = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if d is None:
                return None
            if len(d.shape) == 3:
                d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            return d
        except Exception:
            return None

    @staticmethod
    def _resize_depth(depth, rgb_shape):
        if depth is None or rgb_shape is None:
            return None
        h, w = rgb_shape[:2]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth

    def segment_hand(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_SKIN, self.UPPER_SKIN)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def find_hand(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        hand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand) < self.MIN_HAND_AREA:
            return None
        return hand

    def count_fingers(self, hand_contour):
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            return 0, []
        try:
            defects = cv2.convexityDefects(hand_contour, hull)
        except cv2.error:
            defects = None
        if defects is None:
            return 0, []
        defects = defects.reshape(-1, 4)
        valleys = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i]
            start = tuple(hand_contour[int(s)][0])
            end = tuple(hand_contour[int(e)][0])
            far = tuple(hand_contour[int(f)][0])
            a = np.linalg.norm(np.array(start) - np.array(end))
            b = np.linalg.norm(np.array(start) - np.array(far))
            c = np.linalg.norm(np.array(end) - np.array(far))
            angle = np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c + 1e-6), -1.0, 1.0))
            depth_defect = d / 256.0
            if angle < np.pi / 2 and depth_defect > 15:
                valleys.append(far)
        fingers = min(len(valleys) + 1, 5)
        return fingers, valleys

    def classify_gesture(self, fingers):
        gestures = {
            0: "Fist",
            1: "One",
            2: "Peace",
            3: "Three",
            4: "Four",
            5: "Open Palm",
        }
        return gestures.get(fingers, f"{fingers} fingers")

    def get_distance(self, depth, x, y):
        if depth is None:
            return None
        h, w = depth.shape[:2]
        r = self.DEPTH_SAMPLE_RADIUS
        patch = depth[max(0, y - r):min(h, y + r + 1),
                      max(0, x - r):min(w, x + r + 1)]
        if patch.size == 0:
            return None
        if self.DEPTH_IS_RAW_UINT16:
            valid = patch[patch > 0]
            if valid.size == 0:
                return None
            return float(np.median(valid)) / 1000.0
        return float(np.median(patch)) / 255.0 * self.DEPTH_MAX_METERS

    def loop(self):
        self.init_camera()
        while self.running:
            depth, rgb = self.fetch_frames()
            if rgb is None:
                time.sleep(0.1)
                continue

            depth = self._resize_depth(depth, rgb.shape)
            output = rgb.copy()

            gesture = "No hand"
            distance = None
            handedness = "Unknown"
            emotion = None
            emotion_conf = None

            # Emotion detection
            emo, conf, face_box = self.emotion_detector.detect(rgb)
            if emo is not None:
                emotion = emo
                emotion_conf = conf
                if face_box:
                    x, y, fw, fh = face_box
                    cv2.rectangle(output, (x, y), (x + fw, y + fh), (255, 0, 255), 2)
                    cv2.putText(output, f"{emotion} ({emotion_conf:.2f})",
                                (x, max(y - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Hand gesture detection
            mask = self.segment_hand(rgb)
            hand = self.find_hand(mask)

            if hand is not None:
                cv2.drawContours(output, [hand], -1, (0, 255, 0), 2)
                fingers, valleys = self.count_fingers(hand)
                gesture = self.classify_gesture(fingers)

                M = cv2.moments(hand)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(output, (cx, cy), 7, (255, 0, 0), -1)
                    distance = self.get_distance(depth, cx, cy)

                for v in valleys:
                    cv2.circle(output, v, 5, (0, 255, 255), -1)

                cv2.putText(output, f"Gesture: {gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                dt = f"{distance:.2f}m" if distance else "N/A"
                cv2.putText(output, f"Distance: {dt}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            with self._lock:
                self.latest_frame = output
                self.latest_gesture = gesture
                self.latest_distance = distance
                self.latest_handedness = handedness
                self.latest_emotion = emotion
                self.latest_emotion_conf = emotion_conf

            time.sleep(0.03)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_frame_jpeg(self):
        with self._lock:
            frame = self.latest_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    def get_data(self):
        with self._lock:
            return {
                "gesture": self.latest_gesture,
                "distance": self.latest_distance,
                "handedness": self.latest_handedness,
                "emotion": self.latest_emotion,
                "emotion_conf": self.latest_emotion_conf,
            }
