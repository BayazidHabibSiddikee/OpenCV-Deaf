"""Lightweight emotion detection using OpenCV YuNet face detector + ONNX Runtime.

Downloads a small FER+ emotion model on first use.
Emotions: neutral, happiness, surprise, sadness, anger, disgust, fear
"""

import os
import cv2
import numpy as np
import requests


class EmotionDetector:
    EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

    def __init__(self, model_path=None, yunet_path=None):
        # Look in models directory first, then default location
        self.model_path = model_path or os.path.expanduser(
            "~/.opencv/models/emotion-ferplus-8.onnx"
        )
        if not os.path.exists(self.model_path):
            # Fall back to default location
            self.model_path = model_path or os.path.expanduser(
                "~/.opencv/emotion-ferplus-8.onnx"
            )
        self.yunet_path = yunet_path or os.path.expanduser(
            "~/.opencv/models/face_detection_yunet_2023mar.onnx"
        )
        self.session = None
        self.face_detector = None
        self._init_face_detector()
        self._ensure_emotion_model()

    # ------------------------------------------------------------------ #
    def _init_face_detector(self):
        """Initialize OpenCV YuNet face detector."""
        if not os.path.exists(self.yunet_path):
            print(f"[Emotion] YuNet model not found at {self.yunet_path}")
            return
        try:
            self.face_detector = cv2.FaceDetectorYN_create(
                self.yunet_path, "", (320, 320), score_threshold=0.6
            )
            print("[Emotion] YuNet face detector ready.")
        except Exception as e:
            print(f"[Emotion] Failed to init YuNet: {e}")

    # ------------------------------------------------------------------ #
    def _ensure_emotion_model(self):
        """Download the ONNX emotion model if missing."""
        if os.path.exists(self.model_path):
            self._load_session()
            return

        urls = [
            "https://github.com/microsoft/onnxmodels/raw/main/"
            "validated/emotion_ferplus/emotion_ferplus-8/"
            "model/emotion-ferplus-8.onnx",
            "https://github.com/onnx/models/raw/main/"
            "validated/emotion_ferplus/emotion_ferplus-8/"
            "model/emotion-ferplus-8.onnx",
        ]

        for url in urls:
            try:
                print(f"[Emotion] Downloading model from {url} ...")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                with open(self.model_path, "wb") as f:
                    f.write(resp.content)
                print(f"[Emotion] Model saved to {self.model_path}")
                self._load_session()
                return
            except Exception as e:
                print(f"[Emotion] Failed {url}: {e}")
                continue

        print("[Emotion] Could not download emotion model. Emotion detection disabled.")
        self.session = None

    def _load_session(self):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(self.model_path)
            print("[Emotion] ONNX emotion session loaded.")
        except Exception as e:
            print(f"[Emotion] Failed to load ONNX session: {e}")
            self.session = None

    # ------------------------------------------------------------------ #
    def detect(self, frame):
        """Return (emotion, confidence, face_box) or (None, None, None)."""
        if self.face_detector is None:
            return None, None, None

        h, w = frame.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(frame)

        if faces is None or len(faces) == 0:
            return None, None, None

        # Use the first detected face
        face = faces[0]
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        x = max(0, x)
        y = max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)

        if fw < 48 or fh < 48:
            return None, None, None

        face_roi = frame[y:y + fh, x:x + fw]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
        face_gray = face_gray.astype(np.float32) / 255.0
        input_tensor = face_gray.reshape(1, 1, 64, 64)

        if self.session is None:
            # Face detected but no emotion model yet
            return "face_detected", 1.0, (x, y, fw, fh)

        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            probs = outputs[0][0]
            probs = probs / probs.sum()
            idx = int(np.argmax(probs))
            emotion = self.EMOTIONS[idx]
            confidence = float(probs[idx])
            return emotion, confidence, (x, y, fw, fh)
        except Exception as e:
            print(f"[Emotion] Inference error: {e}")
            return None, None, None
