#!/usr/bin/env python3
"""
MaixSense-A075V + Hugging Face SigLIP-2 (3.5GB) Multi-Hand & Emotion Vision Studio
- Hugging Face Zero-Shot Vision Engine (google/siglip2-so400m-patch14-384, ~3.5GB):
    * Classifies 14+ expressive hand symbols (Fuck Off, Hi/Hello, Rock/Horns, I Love You, etc.)
    * Classifies 8 real-life facial emotions (Happy, Angry, Surprised, Sad, Laughing, etc.)
- MediaPipe 21-Landmark 3D Engine (up to 4 hands simultaneously):
    * Sub-pixel Pinch-to-Draw (touch Thumb + Index tips together to draw at their midpoint)
- Sipeed MaixSense-A075V ToF Depth Sensor:
    * Real-time metric distance (m & mm) for every hand and your face/body
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import time
import struct
import threading
import urllib.request
import cv2
import numpy as np
import requests
from PIL import Image

import torch
from transformers import AutoModel, AutoProcessor

cv2.setNumThreads(2)

# Optional PyAutoGUI for 2-Hand Game Mode
try:
    import pyautogui
    pyautogui.PAUSE = 0.0
    pyautogui.FAILSAFE = True
    HAS_PYAUTOGUI = True
except Exception:
    HAS_PYAUTOGUI = False

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ==================== CONFIGURATION ====================
CAMERA_HOST = "http://192.168.233.1"
HTTP_SESSION = requests.Session()

# Hugging Face SigLIP model ID
# Options:
#   "google/siglip-base-patch16-224"      (~800 MB, fast CPU inference, reliable download)
#   "google/siglip2-base-patch16-224"     (~1.5 GB, newer SigLIP2 architecture)
#   "google/siglip2-so400m-patch14-384"   (~3.5 GB flagship, requires high RAM)
HF_MODEL_ID = os.getenv("HF_SIGLIP_MODEL", "google/siglip-base-patch16-224")

HAND_PALETTE = [
    ("Cyan", (255, 240, 0)),
    ("Green", (60, 255, 80)),
    ("Gold", (0, 215, 255)),
    ("Magenta", (255, 70, 255)),
]

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),
)

# ==================== EXPRESSIVE GESTURE & EMOTION PROMPTS ====================
GESTURE_PROMPTS = {
    "Hi / Hello": "a close-up photo of an open human hand with all five fingers spread wide waving hello",
    "Fuck Off": "a close-up photo of a human hand raising only the middle finger upward in an offensive gesture with other fingers curled down",
    "Fist [ERASE]": "a close-up photo of a tightly closed human fist with all fingers curled into the palm",
    "Peace / Victory": "a close-up photo of a human hand making a V peace sign with index and middle fingers raised",
    "Rock / Horns": "a close-up photo of a human hand making the rock and roll devil horns sign with index and pinky fingers raised",
    "I Love You": "a close-up photo of a human hand making the I love you sign with thumb, index, and pinky fingers extended",
    "Yes / Like": "a close-up photo of a human hand giving a thumbs up approval gesture",
    "No / Dislike": "a close-up photo of a human hand giving a thumbs down disapproval gesture",
    "OK Sign": "a close-up photo of a human hand making an OK sign with thumb and index forming a circle and three fingers up",
    "Call Me": "a close-up photo of a human hand making a telephone call me shaka sign with thumb and pinky extended",
    "Finger Gun": "a close-up photo of a human hand shaped like a pistol gun with thumb up and index finger pointing",
    "Point / One": "a close-up photo of a human hand pointing upward with only the index finger extended",
    "Salute": "a close-up photo of a human hand held flat with fingers together in a military salute",
    "Stop": "a close-up photo of a flat open palm facing forward signaling stop",
}

EMOTION_PROMPTS = {
    "Happy": "a close-up portrait photo of a person with a happy smiling facial expression",
    "Laughing": "a close-up portrait photo of a person laughing joyfully with an open mouth smile",
    "Angry": "a close-up portrait photo of a person with an angry frowning annoyed facial expression",
    "Surprised": "a close-up portrait photo of a person with a shocked surprised expression and wide eyes",
    "Sad": "a close-up portrait photo of a person with a sad sorrowful downbeat facial expression",
    "Disgusted": "a close-up portrait photo of a person with a disgusted grimacing facial expression",
    "Fearful": "a close-up portrait photo of a person with a scared fearful anxious facial expression",
    "Neutral": "a close-up portrait photo of a person with a calm neutral relaxed facial expression",
}


# ==================== HUGGING FACE ASYNC VISION ENGINE ====================

class HuggingFaceVisionWorker:
    """
    Runs google/siglip2-so400m-patch14-384 (~3.5GB) in a background thread.
    Pre-caches text embeddings at startup so live batches of [hands..., face] run fast.
    """
    def __init__(self, model_id=HF_MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[HF] Loading {model_id} on {self.device.upper()} ({self.dtype})...")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        self.gesture_labels = list(GESTURE_PROMPTS.keys())
        self.emotion_labels = list(EMOTION_PROMPTS.keys())

        # Pre-compute normalized text embeddings ONCE at startup
        with torch.no_grad():
            g_in = self.processor(
                text=list(GESTURE_PROMPTS.values()),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            g_emb = self.model.get_text_features(**g_in)
            self.gesture_embeds = g_emb / g_emb.norm(p=2, dim=-1, keepdim=True)

            e_in = self.processor(
                text=list(EMOTION_PROMPTS.values()),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            e_emb = self.model.get_text_features(**e_in)
            self.emotion_embeds = e_emb / e_emb.norm(p=2, dim=-1, keepdim=True)

        print(f"[HF] Ready! RAM footprint ~3.5GB. Zero-shot prompts active.")

        self._lock = threading.Lock()
        self._pending_batch = None
        self._hand_results = {}
        self._emotion_result = (None, 0.0)
        self._inference_ms = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def submit(self, hand_crops, face_crop):
        with self._lock:
            self._pending_batch = (hand_crops, face_crop)

    def get_results(self):
        with self._lock:
            return dict(self._hand_results), self._emotion_result, self._inference_ms

    def stop(self):
        self._running = False

    def _worker_loop(self):
        while self._running:
            with self._lock:
                batch = self._pending_batch
                self._pending_batch = None

            if batch is None:
                time.sleep(0.01)
                continue

            hand_crops, face_crop = batch
            if not hand_crops and face_crop is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            pil_images = []
            hand_slots = list(hand_crops.keys())
            for s in hand_slots:
                pil_images.append(hand_crops[s])
            has_face = face_crop is not None
            if has_face:
                pil_images.append(face_crop)

            try:
                inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
                if self.dtype == torch.float16:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

                with torch.no_grad():
                    img_emb = self.model.get_image_features(**inputs)
                    img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
                    scale = self.model.logit_scale.exp()
                    bias = self.model.logit_bias

                    new_hand_res = {}
                    for i, slot in enumerate(hand_slots):
                        logits = (img_emb[i : i + 1] @ self.gesture_embeds.T) * scale + bias
                        probs = torch.sigmoid(logits)[0]
                        best_idx = int(torch.argmax(probs).item())
                        new_hand_res[slot] = (
                            self.gesture_labels[best_idx],
                            float(probs[best_idx].item()),
                        )

                    new_emo_res = self._emotion_result
                    if has_face:
                        f_emb = img_emb[-1:]
                        logits = (f_emb @ self.emotion_embeds.T) * scale + bias
                        probs = torch.sigmoid(logits)[0]
                        best_idx = int(torch.argmax(probs).item())
                        new_emo_res = (
                            self.emotion_labels[best_idx],
                            float(probs[best_idx].item()),
                        )

                dt_ms = (time.time() - t0) * 1000.0
                with self._lock:
                    self._hand_results = new_hand_res
                    self._emotion_result = new_emo_res
                    self._inference_ms = dt_ms
            except Exception as e:
                print(f"[HF Worker] Inference warning: {e}")


# ==================== MEDIAPIPE & YUNET HELPERS ====================

def ensure_hand_landmarker_model():
    model_dir = os.path.expanduser("~/.opencv/models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "hand_landmarker.task")
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        urllib.request.urlretrieve(url, model_path)
    return model_path


def init_yunet():
    yunet_path = os.path.expanduser("~/.opencv/models/face_detection_yunet_2023mar.onnx")
    if os.path.exists(yunet_path):
        try:
            return cv2.FaceDetectorYN_create(yunet_path, "", (320, 240), score_threshold=0.55)
        except Exception:
            pass
    return None


def init_camera(host=CAMERA_HOST):
    try:
        cfg = struct.pack("<BBBBBBBBi", 1, 0, 255, 0, 2, 7, 1, 0, 0)
        HTTP_SESSION.post(f"{host}/set_cfg", data=cfg, timeout=2)
    except Exception:
        pass


def fetch_a075v_frames(host=CAMERA_HOST):
    try:
        resp = HTTP_SESSION.get(f"{host}/getdeep", timeout=1.5)
        data = resp.content
        if len(data) < 36:
            return None, None
        _, deep_mode, _, ir_mode, status_mode, _, rgb_mode, _, _ = struct.unpack("<BBBBBBBBi", data[16:28])
        deep_data_size, rgb_data_size = struct.unpack("<ii", data[28:36])
        payload = data[36:]

        deepth_size = (320 * 240 * 2) >> deep_mode
        ir_size = (320 * 240 * 2) >> ir_mode
        status_size = (320 * 240 // 8) * (
            16 if status_mode == 0 else 2 if status_mode == 1 else 8 if status_mode == 2 else 1
        )

        depth = None
        if deepth_size > 0 and len(payload) >= deepth_size:
            dtype = np.uint16 if deep_mode == 0 else np.uint8
            depth = np.frombuffer(payload[:deepth_size], dtype=dtype).reshape((240, 320)).copy()

        rgb_offset = deepth_size + ir_size + status_size
        rgb_bytes = payload[rgb_offset : rgb_offset + rgb_data_size]
        rgb = None
        if rgb_data_size > 0 and len(rgb_bytes) == rgb_data_size and rgb_mode == 1:
            rgb = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return depth, rgb
    except Exception:
        return None, None


def enhance_contrast(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def sample_distance_m(depth_map, cx, cy, r=6):
    if depth_map is None:
        return None
    h, w = depth_map.shape[:2]
    cx, cy = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
    patch = depth_map[max(0, cy - r) : min(h, cy + r + 1), max(0, cx - r) : min(w, cx + r + 1)]
    if patch.size == 0:
        return None
    if depth_map.dtype == np.uint16:
        valid = patch[(patch > 80) & (patch < 4000)]
        return float(np.median(valid)) / 1000.0 if valid.size > 0 else None
    valid = patch[patch > 0]
    return float(np.median(valid)) / 255.0 * 2.0 if valid.size > 0 else None


def crop_hand_pil(bgr_frame, lm_list, pad_ratio=0.30):
    h, w = bgr_frame.shape[:2]
    xs = [l.x * w for l in lm_list]
    ys = [l.y * h for l in lm_list]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bw, bh = max(x1 - x0, 20.0), max(y1 - y0, 20.0)
    pad_x, pad_y = bw * pad_ratio, bh * pad_ratio

    ix0 = max(0, int(x0 - pad_x))
    iy0 = max(0, int(y0 - pad_y))
    ix1 = min(w, int(x1 + pad_x))
    iy1 = min(h, int(y1 + pad_y))
    if (ix1 - ix0) < 32 or (iy1 - iy0) < 32:
        return None
    crop_rgb = cv2.cvtColor(bgr_frame[iy0:iy1, ix0:ix1], cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def extract_pinch_and_fallback_gesture(lm_list, vw, vh):
    pts = np.array([[l.x, l.y] for l in lm_list], dtype=np.float32)
    wrist = pts[0]
    middle_mcp = pts[9]
    pinky_mcp = pts[17]
    palm_scale = max(float(np.linalg.norm(middle_mcp - wrist)), 1e-4)

    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    states = [False] * 5
    states[0] = np.linalg.norm(pts[4] - pinky_mcp) > np.linalg.norm(pts[3] - pinky_mcp) * 1.15
    for i in range(1, 5):
        states[i] = np.linalg.norm(pts[tips[i]] - wrist) > np.linalg.norm(pts[pips[i]] - wrist) * 1.12

    thumb, index, middle, ring, pinky = states
    count = sum(states)

    pinch_ratio = float(np.linalg.norm(pts[4] - pts[8])) / palm_scale
    index_extended_from_palm = np.linalg.norm(pts[8] - wrist) > np.linalg.norm(pts[5] - wrist) * 1.08
    is_pinching = (pinch_ratio < 0.28) and (index_extended_from_palm or middle or ring or pinky)

    draw_pt = (
        int(((pts[4][0] + pts[8][0]) * 0.5) * vw),
        int(((pts[4][1] + pts[8][1]) * 0.5) * vh),
    )
    palm_pt = (
        int(((pts[0][0] + pts[5][0] + pts[17][0]) / 3.0) * vw),
        int(((pts[0][1] + pts[5][1] + pts[17][1]) / 3.0) * vh),
    )

    if is_pinching:
        fallback = "Pinch [DRAWING]"
    elif middle and not index and not ring and not pinky:
        fallback = "Fuck Off"
    elif count == 0:
        fallback = "Fist [ERASE]"
    elif count == 5:
        fallback = "Hi / Hello"
    elif index and middle and not ring and not pinky:
        fallback = "Peace / Victory"
    else:
        fallback = f"{count} Fingers"

    return is_pinching, draw_pt, palm_pt, fallback


# ==================== MAIN STUDIO APPLICATION ====================

def main():
    print("=================================================================")
    print(" MaixSense-A075V + Hugging Face SigLIP-2 (3.5GB) Vision Studio")
    print(" Controls:")
    print("   Touch Thumb + Index  -> DRAW smooth neon lines at midpoint")
    print("   Separate Fingers     -> HOVER cursor (no ink)")
    print("   Closed Fist          -> ERASE canvas")
    print("   Hotkeys: [d]Draw On/Off  [c]Clear  [g]2H-Game  [m]Flip  [q]Quit")
    print("=================================================================")

    hf_worker = HuggingFaceVisionWorker(HF_MODEL_ID)
    yunet = init_yunet()

    task_path = ensure_hand_landmarker_model()
    base_opts = mp_tasks.BaseOptions(model_asset_path=task_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=4,
        min_hand_detection_confidence=0.40,
        min_hand_presence_confidence=0.40,
        min_tracking_confidence=0.40,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    init_camera()

    draw_enabled = True
    game_enabled = False
    mirror = True

    canvas_layer = None
    smooth_pts = {}
    prev_draw_pts = {}
    last_game_action = 0.0
    last_ts_ms = 0
    prev_time = time.time()

    win_name = "MaixSense-A075V + HF SigLIP-2 Studio"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            depth, rgb = fetch_a075v_frames()
            if rgb is None:
                time.sleep(0.02)
                continue

            if mirror:
                rgb = cv2.flip(rgb, 1)
                if depth is not None:
                    depth = cv2.flip(depth, 1)

            vh, vw = rgb.shape[:2]
            if canvas_layer is None or canvas_layer.shape[:2] != (vh, vw):
                canvas_layer = np.zeros_like(rgb)

            raw_depth = depth
            depth_resized = (
                cv2.resize(depth, (vw, vh), interpolation=cv2.INTER_NEAREST)
                if (depth is not None and depth.shape[:2] != (vh, vw))
                else depth
            )

            enhanced_rgb = enhance_contrast(rgb)
            output = enhanced_rgb.copy()

            # 1. Detect Face Box (YuNet) + Sample Face/Person Metric Distance
            face_box = None
            face_crop_pil = None
            person_dist_m = None

            if yunet is not None:
                yunet.setInputSize((vw, vh))
                _, faces = yunet.detect(enhanced_rgb)
                if faces is not None and len(faces) > 0:
                    f = max(faces, key=lambda x: x[2] * x[3])
                    fx, fy, fw, fh = max(0, int(f[0])), max(0, int(f[1])), int(f[2]), int(f[3])
                    fw, fh = min(fw, vw - fx), min(fh, vh - fy)
                    if fw >= 40 and fh >= 40:
                        face_box = (fx, fy, fw, fh)
                        person_dist_m = sample_distance_m(depth_resized, fx + fw // 2, fy + fh // 2, r=10)
                        face_crop_pil = Image.fromarray(
                            cv2.cvtColor(enhanced_rgb[fy : fy + fh, fx : fx + fw], cv2.COLOR_BGR2RGB)
                        )

            if person_dist_m is None:
                person_dist_m = sample_distance_m(depth_resized, vw // 2, vh // 2, r=15)

            # 2. Detect up to 4 Hands via MediaPipe 3D Landmarks
            rgb_contig = np.ascontiguousarray(cv2.cvtColor(enhanced_rgb, cv2.COLOR_BGR2RGB))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contig)
            ts_ms = int(time.monotonic() * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            mp_res = landmarker.detect_for_video(mp_img, ts_ms)
            hands_list = []

            if mp_res.hand_landmarks:
                for idx, lm_list in enumerate(mp_res.hand_landmarks):
                    side = "Hand"
                    if mp_res.handedness and idx < len(mp_res.handedness):
                        side = mp_res.handedness[idx][0].category_name
                    is_pinching, draw_pt, palm_pt, fallback_g = extract_pinch_and_fallback_gesture(lm_list, vw, vh)
                    dist_m = sample_distance_m(depth_resized, palm_pt[0], palm_pt[1], r=6)
                    hands_list.append({
                        "side": side,
                        "is_pinching": is_pinching,
                        "draw_pt": draw_pt,
                        "palm": palm_pt,
                        "fallback": fallback_g,
                        "dist_m": dist_m,
                        "landmarks": lm_list,
                    })
                hands_list.sort(key=lambda h: h["palm"][0])

            # 3. Submit Hand Crops + Face Crop to Async Hugging Face SigLIP-2 Worker
            hand_crops = {}
            for idx, h in enumerate(hands_list):
                h["id"] = idx + 1
                crop = crop_hand_pil(enhanced_rgb, h["landmarks"])
                if crop is not None:
                    hand_crops[idx] = crop

            hf_worker.submit(hand_crops, face_crop_pil)
            hf_hand_preds, (hf_emo, hf_emo_conf), hf_ms = hf_worker.get_results()

            # Render Face Emotion Box
            if face_box is not None:
                fx, fy, fw, fh = face_box
                cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh), (255, 70, 255), 2)
                emo_txt = f"{hf_emo} ({int(hf_emo_conf * 100)}%)" if hf_emo else "Analyzing Face..."
                if person_dist_m:
                    emo_txt += f" | {person_dist_m:.2f}m"
                cv2.putText(output, emo_txt, (fx, max(20, fy - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 110, 255), 1)

            # 4. Render Hands, HF Labels, & Pinch-to-Draw Canvas
            active_slots = set()
            for idx, h in enumerate(hands_list):
                active_slots.add(idx)
                _, h_color = HAND_PALETTE[idx % len(HAND_PALETTE)]
                palm_pt = h["palm"]
                draw_pt = h["draw_pt"]

                if h["is_pinching"]:
                    gesture = "Pinch [DRAWING]"
                    conf_str = ""
                elif idx in hf_hand_preds and hf_hand_preds[idx][1] >= 0.22:
                    lbl, conf = hf_hand_preds[idx]
                    gesture = lbl
                    conf_str = f" {int(conf * 100)}%"
                else:
                    gesture = h["fallback"]
                    conf_str = ""

                h["gesture"] = gesture

                pts = [(int(l.x * vw), int(l.y * vh)) for l in h["landmarks"]]
                for a, b in HAND_EDGES:
                    cv2.line(output, pts[a], pts[b], h_color, 2, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(output, p, 3, (255, 255, 255), -1)
                cv2.circle(output, pts[4], 6, (0, 255, 255), -1)
                cv2.circle(output, pts[8], 6, (0, 255, 255), -1)

                tag = f"H{h['id']} ({h['side']}): {gesture}{conf_str}"
                if h["dist_m"]:
                    tag += f" | {h['dist_m']:.2f}m"
                cv2.putText(output, tag, (max(8, palm_pt[0] - 80), max(24, palm_pt[1] - 25)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, h_color, 1)

                if draw_enabled:
                    if idx not in smooth_pts:
                        smooth_pts[idx] = np.array(draw_pt, dtype=np.float32)
                    else:
                        smooth_pts[idx] = 0.65 * np.array(draw_pt, dtype=np.float32) + 0.35 * smooth_pts[idx]
                    cur_pt = (int(smooth_pts[idx][0]), int(smooth_pts[idx][1]))

                    if h["is_pinching"]:
                        cv2.circle(output, cur_pt, 9, h_color, -1)
                        cv2.circle(output, cur_pt, 13, (255, 255, 255), 2)
                        if idx in prev_draw_pts and prev_draw_pts[idx] is not None:
                            cv2.line(canvas_layer, prev_draw_pts[idx], cur_pt, h_color, 5, cv2.LINE_AA)
                        prev_draw_pts[idx] = cur_pt
                    elif "Fist" in gesture:
                        cv2.circle(output, palm_pt, 30, (0, 0, 255), 2)
                        cv2.circle(canvas_layer, palm_pt, 30, (0, 0, 0), -1)
                        prev_draw_pts[idx] = None
                    else:
                        cv2.circle(output, cur_pt, 7, (210, 210, 210), 1)
                        prev_draw_pts[idx] = None

            for old_idx in list(prev_draw_pts.keys()):
                if old_idx not in active_slots:
                    prev_draw_pts.pop(old_idx, None)
                    smooth_pts.pop(old_idx, None)

            c_gray = cv2.cvtColor(canvas_layer, cv2.COLOR_BGR2GRAY)
            _, c_mask = cv2.threshold(c_gray, 5, 255, cv2.THRESH_BINARY)
            output[c_mask > 0] = cv2.addWeighted(output, 0.25, canvas_layer, 0.75, 0)[c_mask > 0]

            # 5. Two-Hand Game Controller Mode
            if game_enabled and HAS_PYAUTOGUI and hands_list:
                now = time.time()
                if (now - last_game_action) > 0.35:
                    for h in hands_list:
                        is_left = (h["side"] == "Left") or (len(hands_list) > 1 and h["id"] == 1)
                        g = h["gesture"]
                        if is_left:
                            if "Fist" in g:
                                pyautogui.press("down")
                                last_game_action = now
                            elif "Yes" in g:
                                pyautogui.press("left")
                                last_game_action = now
                            elif "Peace" in g:
                                pyautogui.press("right")
                                last_game_action = now
                        else:
                            if "Hi" in g or "Stop" in g:
                                pyautogui.press("space")
                                last_game_action = now
                            elif h["is_pinching"]:
                                pyautogui.click()
                                last_game_action = now

            # 6. Telemetry Dashboard
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            pw = 375
            ui = np.zeros((vh, vw + pw, 3), dtype=np.uint8)
            ui[:, :vw] = output
            panel = ui[:, vw:]
            panel[:] = (22, 22, 28)
            cv2.line(ui, (vw, 0), (vw, vh), (55, 55, 70), 2)

            cv2.putText(panel, "A075V + HF SIGLIP-2 STUDIO", (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.56, (0, 220, 255), 1)
            cv2.putText(panel, f"HF LATENCY: {hf_ms:.0f} ms | HANDS: {len(hands_list)}", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 235, 140), 1)
            cv2.line(panel, (16, 58), (pw - 16, 58), (55, 55, 65), 1)

            y_cursor = 82
            if not hands_list:
                cv2.putText(panel, "Show 1-4 hands to camera...", (16, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 135), 1)
            else:
                for slot in range(4):
                    if slot < len(hands_list):
                        h = hands_list[slot]
                        _, col = HAND_PALETTE[slot % len(HAND_PALETTE)]
                        d_str = f"{h['dist_m']:.2f}m" if h["dist_m"] else "--m"
                        line = f"H{h['id']} [{h['side']}]: {h['gesture']} | {d_str}"
                        cv2.circle(panel, (22, y_cursor - 4), 5, col, -1)
                        cv2.putText(panel, line, (34, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
                    y_cursor += 24

            cv2.line(panel, (16, 182), (pw - 16, 182), (55, 55, 65), 1)

            p_str = f"{person_dist_m:.2f} m ({int(person_dist_m*1000)} mm)" if person_dist_m else "-- m"
            cv2.putText(panel, f"PERSON DIST: {p_str}", (16, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 175, 75), 1)

            emo_str = f"{hf_emo} ({int(hf_emo_conf * 100)}%)" if hf_emo else "No Face Detected"
            cv2.putText(panel, f"HF EMOTION: {emo_str}", (16, 232), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 120, 255), 1)

            cv2.putText(panel, f"PINCH-DRAW[d]: {'ON' if draw_enabled else 'OFF'}", (16, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 240, 255) if draw_enabled else (110, 110, 110), 1)
            cv2.putText(panel, f"2H-GAME[g]: {'ON' if game_enabled else 'OFF'}", (200, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 255, 120) if game_enabled else (110, 110, 110), 1)

            tw, th, ty = 160, 102, 290
            cv2.putText(panel, "TOF DEPTH MAP", (16, 283), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
            if raw_depth is not None:
                dn = np.clip(raw_depth, 100, 2000) if raw_depth.dtype == np.uint16 else raw_depth
                if raw_depth.dtype == np.uint16:
                    dn = ((dn - 100) / 1900.0 * 255).astype(np.uint8)
                panel[ty : ty + th, 16 : 16 + tw] = cv2.resize(cv2.applyColorMap(dn, cv2.COLORMAP_JET), (tw, th))

            cv2.putText(panel, "PINCH AIR CANVAS", (192, 283), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
            panel[ty : ty + th, 192 : 192 + tw] = cv2.resize(canvas_layer, (tw, th))

            cv2.putText(panel, f"VIDEO FPS: {fps:.1f} | [c]Clear [m]Flip [q]Quit",
                        (16, 458), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (165, 165, 180), 1)

            cv2.imshow(win_name, ui)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("c"):
                canvas_layer[:] = 0
            elif k == ord("d"):
                draw_enabled = not draw_enabled
            elif k == ord("g"):
                game_enabled = not game_enabled
            elif k == ord("m"):
                mirror = not mirror

    finally:
        hf_worker.stop()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
