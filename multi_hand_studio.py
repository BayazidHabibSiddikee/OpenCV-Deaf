#!/usr/bin/env python3
"""
MaixSense-A075V Multi-Hand Expressive Gesture & Pinch-Draw Studio
- Pinch-to-Draw: Touch Thumb + Index fingertips together to draw at their exact midpoint
- 14 Expressive Gestures: Hi/Hello, Fuck Off, Rock/Horns, I Love You, Call Me, Yes/Like, No/Dislike, Peace, Fist, etc.
- Multi-Hand (1-4 Hands) + Metric Distance (Hand & Person) + Live Emotion + 2-Hand Game Control
"""

import os
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import time
import struct
import urllib.request
import cv2
import numpy as np
import requests

cv2.setNumThreads(2)

try:
    import pyautogui
    pyautogui.PAUSE = 0.0
    pyautogui.FAILSAFE = True
    HAS_PYAUTOGUI = True
except Exception:
    HAS_PYAUTOGUI = False

HAS_MP_TASKS = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MP_TASKS = True
except Exception as e:
    print(f"[Info] MediaPipe Tasks unavailable ({e}), using 3D ToF Multi-Blob Tracker.")

CAMERA_HOST = "http://192.168.233.1"
HTTP_SESSION = requests.Session()

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


# ==================== MODEL SETUP ====================

def ensure_hand_landmarker_model():
    model_dir = os.path.expanduser("~/.opencv/models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "hand_landmarker.task")
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        print(f"[Model] Downloading hand_landmarker.task to {model_path} ...")
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            print(f"[Model] Download failed ({e}).")
            return None
    return model_path


class LightweightEmotionDetector:
    """Native cv2.dnn YuNet + FER+ detector (zero onnxruntime RAM overhead)."""
    EMOTIONS = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear"]

    def __init__(self):
        self.face_det = None
        self.emo_net = None
        yunet_path = os.path.expanduser("~/.opencv/models/face_detection_yunet_2023mar.onnx")
        emo_path = os.path.expanduser("~/.opencv/models/emotion-ferplus-8.onnx")
        if not os.path.exists(emo_path):
            emo_path = os.path.expanduser("~/.opencv/emotion-ferplus-8.onnx")

        if os.path.exists(yunet_path):
            try:
                self.face_det = cv2.FaceDetectorYN_create(yunet_path, "", (320, 240), score_threshold=0.55)
            except Exception:
                pass
        if os.path.exists(emo_path):
            try:
                self.emo_net = cv2.dnn.readNetFromONNX(emo_path)
            except Exception:
                pass

    def detect(self, frame):
        if self.face_det is None:
            return None, None, None
        h, w = frame.shape[:2]
        self.face_det.setInputSize((w, h))
        _, faces = self.face_det.detect(frame)
        if faces is None or len(faces) == 0:
            return None, None, None

        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = max(0, int(face[0])), max(0, int(face[1])), int(face[2]), int(face[3])
        fw, fh = min(fw, w - x), min(fh, h - y)
        if fw < 40 or fh < 40:
            return None, None, None

        if self.emo_net is None:
            return "Face", 1.0, (x, y, fw, fh)

        try:
            roi = cv2.cvtColor(frame[y : y + fh, x : x + fw], cv2.COLOR_BGR2GRAY)
            blob = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(1, 1, 64, 64)
            self.emo_net.setInput(blob)
            logits = self.emo_net.forward()[0][:7]
            exp_s = np.exp(logits - np.max(logits))
            probs = exp_s / np.sum(exp_s)
            idx = int(np.argmax(probs))
            return self.EMOTIONS[idx], float(probs[idx]), (x, y, fw, fh)
        except Exception:
            return "Face", 1.0, (x, y, fw, fh)


# ==================== CAMERA & DEPTH HELPERS ====================

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


# ==================== 14-SYMBOL EXPRESSIVE GESTURE & PINCH CLASSIFIER ====================

def classify_landmarks(lm_list, vw, vh):
    """
    Scale-invariant 21-landmark classifier supporting:
    - Pinch [DRAWING] (Thumb tip #4 + Index tip #8 touching)
    - Fuck Off (Middle finger #12 extended alone)
    - Hi / Hello (Open Palm)
    - Rock / Horns, I Love You, Call Me, Yes/Like, No/Dislike, Peace, Finger Gun, Fist, etc.
    Returns: (gesture_label, finger_count, palm_xy, draw_xy, is_pinching)
    """
    pts = np.array([[l.x, l.y] for l in lm_list], dtype=np.float32)
    wrist = pts[0]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    pinky_mcp = pts[17]

    palm_scale = max(float(np.linalg.norm(middle_mcp - wrist)), 1e-4)

    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    states = [False] * 5
    states[0] = np.linalg.norm(pts[4] - pinky_mcp) > np.linalg.norm(pts[3] - pinky_mcp) * 1.15

    for i in range(1, 5):
        tip_dist = np.linalg.norm(pts[tips[i]] - wrist)
        pip_dist = np.linalg.norm(pts[pips[i]] - wrist)
        states[i] = tip_dist > pip_dist * 1.12

    thumb, index, middle, ring, pinky = states
    count = sum(states)

    pinch_dist = float(np.linalg.norm(pts[4] - pts[8]))
    pinch_ratio = pinch_dist / palm_scale

    index_not_fist = np.linalg.norm(pts[8] - wrist) > np.linalg.norm(pts[5] - wrist) * 1.08
    is_pinching = (pinch_ratio < 0.28) and (index_not_fist or middle or ring or pinky)

    draw_x = int(((pts[4][0] + pts[8][0]) * 0.5) * vw)
    draw_y = int(((pts[4][1] + pts[8][1]) * 0.5) * vh)
    cx = int(((pts[0][0] + pts[5][0] + pts[17][0]) / 3.0) * vw)
    cy = int(((pts[0][1] + pts[5][1] + pts[17][1]) / 3.0) * vh)

    if is_pinching:
        gesture = "Pinch [DRAWING]"
    elif middle and not index and not ring and not pinky:
        gesture = "Fuck Off"
    elif index and pinky and not middle and not ring:
        if thumb:
            gesture = "I Love You"
        else:
            gesture = "Rock / Horns"
    elif thumb and pinky and not index and not middle and not ring:
        gesture = "Call Me"
    elif thumb and not index and not middle and not ring and not pinky:
        gesture = "Yes / Like" if pts[4][1] < pts[2][1] else "No / Dislike"
    elif thumb and index and not middle and not ring and not pinky:
        gesture = "Finger Gun"
    elif index and middle and not ring and not pinky:
        gesture = "Peace / Victory"
    elif index and not middle and not ring and not pinky:
        gesture = "Point / One"
    elif count == 0:
        gesture = "Fist [ERASE]"
    elif count == 5:
        gesture = "Hi / Hello"
    elif count == 3:
        gesture = "Three"
    elif count == 4:
        gesture = "Four"
    else:
        gesture = f"{count} Fingers"

    return gesture, count, (cx, cy), (draw_x, draw_y), is_pinching


# ==================== 3D TOF DEPTH FALLBACK ====================

def detect_multi_hands_from_depth(depth_resized, max_dist_mm=1800):
    if depth_resized is None or depth_resized.dtype != np.uint16:
        return [], np.zeros((480, 640), dtype=np.uint8)

    fg = ((depth_resized > 100) & (depth_resized < max_dist_mm)).astype(np.uint8) * 255
    d_float = depth_resized.astype(np.float32)
    gx = np.abs(cv2.Sobel(d_float, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(d_float, cv2.CV_32F, 0, 1, ksize=3))
    edges = ((gx + gy) > 260.0).astype(np.uint8) * 255
    fg = cv2.bitwise_and(fg, cv2.bitwise_not(edges))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg)
    detected_hands = []

    for lbl in range(1, num_labels):
        px_area = stats[lbl, cv2.CC_STAT_AREA]
        if px_area < 450:
            continue
        cx, cy = int(centroids[lbl][0]), int(centroids[lbl][1])
        dist_m = sample_distance_m(depth_resized, cx, cy, r=5)
        if dist_m is None:
            continue

        real_cm2 = px_area * (dist_m ** 2) * 0.065
        if not (55.0 <= real_cm2 <= 420.0):
            continue

        comp_mask = (labels == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])

        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        hull_pts = cv2.convexHull(cnt, returnPoints=True)
        solidity = float(px_area) / (cv2.contourArea(hull_pts) + 1e-5)
        valleys = 0
        if hull_idx is not None and len(hull_idx) >= 4:
            try:
                defects = cv2.convexityDefects(cnt, hull_idx)
                if defects is not None:
                    _, _, _, bh = cv2.boundingRect(cnt)
                    for s, e, f, d in defects.reshape(-1, 4):
                        start, end, far = cnt[int(s)][0], cnt[int(e)][0], cnt[int(f)][0]
                        a = np.linalg.norm(start - end)
                        b = np.linalg.norm(start - far)
                        c = np.linalg.norm(end - far)
                        ang = np.arccos(np.clip((b*b + c*c - a*a) / (2*b*c + 1e-6), -1.0, 1.0))
                        if ang < np.deg2rad(85) and (d / 256.0) > max(8.0, bh * 0.12):
                            valleys += 1
            except cv2.error:
                pass

        if valleys == 0:
            gesture = "Fist [ERASE]" if solidity > 0.82 else "Pinch [DRAWING]"
            fingers = 0 if solidity > 0.82 else 1
        else:
            fingers = min(valleys + 1, 5)
            gesture = {1: "Point / One", 2: "Peace / Victory", 3: "Three", 4: "Four", 5: "Hi / Hello"}.get(fingers, "Hi / Hello")

        detected_hands.append({
            "id": len(detected_hands) + 1,
            "side": f"Hand {len(detected_hands) + 1}",
            "gesture": gesture,
            "fingers": fingers,
            "palm": (cx, cy),
            "draw_pt": topmost,
            "is_drawing": (gesture == "Pinch [DRAWING]"),
            "dist_m": dist_m,
            "landmarks": None,
            "contour": cnt,
        })
        if len(detected_hands) >= 4:
            break

    detected_hands.sort(key=lambda h: h["palm"][0])
    for idx, h in enumerate(detected_hands):
        h["id"] = idx + 1
    return detected_hands, fg


# ==================== MAIN STUDIO LOOP ====================

def main():
    print("=================================================================")
    print(" MaixSense-A075V Expressive Multi-Hand & Pinch-Draw Studio")
    print(" Controls:")
    print("   Touch Thumb + Index  -> DRAW on screen")
    print("   Separate Fingers     -> HOVER cursor (no ink)")
    print("   Closed Fist          -> ERASE canvas")
    print("   Hotkeys: [d]Draw On/Off  [c]Clear  [g]Game Ctrl  [ [ / ] ]Range  [q]Quit")
    print("=================================================================")

    init_camera()
    emo_detector = LightweightEmotionDetector()

    landmarker = None
    if HAS_MP_TASKS:
        task_path = ensure_hand_landmarker_model()
        if task_path:
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

    draw_enabled = True
    game_enabled = False
    mirror = True
    max_hand_mm = 1800

    canvas_layer = None
    smooth_pts = {}
    prev_draw_pts = {}

    last_game_action = 0.0
    frame_idx = 0
    last_ts_ms = 0
    cached_emo, cached_conf, cached_face = None, None, None
    prev_time = time.time()

    win_name = "MaixSense-A075V - Expressive Multi-Hand Studio"
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

            # 1. Emotion & Person Distance
            frame_idx += 1
            if frame_idx % 4 == 0:
                cached_emo, cached_conf, cached_face = emo_detector.detect(enhanced_rgb)

            person_dist_m = None
            if cached_face is not None:
                fx, fy, fw, fh = cached_face
                person_dist_m = sample_distance_m(depth_resized, fx + fw // 2, fy + fh // 2, r=10)
                cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh), (255, 70, 255), 2)
                emo_lbl = f"{cached_emo} ({int((cached_conf or 0)*100)}%)"
                if person_dist_m:
                    emo_lbl += f" | {person_dist_m:.2f}m"
                cv2.putText(output, emo_lbl, (fx, max(20, fy - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 110, 255), 1)
            else:
                person_dist_m = sample_distance_m(depth_resized, vw // 2, vh // 2, r=15)

            # 2. Multi-Hand Detection
            hands_list = []
            tracker_mode = "TOF MULTI-BLOB"
            debug_mask = np.zeros((vh, vw), dtype=np.uint8)

            if landmarker is not None:
                rgb_contig = np.ascontiguousarray(cv2.cvtColor(enhanced_rgb, cv2.COLOR_BGR2RGB))
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contig)
                ts_ms = int(time.monotonic() * 1000)
                if ts_ms <= last_ts_ms:
                    ts_ms = last_ts_ms + 1
                last_ts_ms = ts_ms

                mp_res = landmarker.detect_for_video(mp_img, ts_ms)
                if mp_res.hand_landmarks:
                    tracker_mode = f"MP-4HAND ({len(mp_res.hand_landmarks)} ACTIVE)"
                    for idx, lm_list in enumerate(mp_res.hand_landmarks):
                        side = "Hand"
                        if mp_res.handedness and idx < len(mp_res.handedness):
                            side = mp_res.handedness[idx][0].category_name
                        gesture, fingers, palm_pt, draw_pt, is_pinching = classify_landmarks(lm_list, vw, vh)
                        dist_m = sample_distance_m(depth_resized, palm_pt[0], palm_pt[1], r=6)
                        hands_list.append({
                            "id": idx + 1,
                            "side": side,
                            "gesture": gesture,
                            "fingers": fingers,
                            "palm": palm_pt,
                            "draw_pt": draw_pt,
                            "is_drawing": is_pinching,
                            "dist_m": dist_m,
                            "landmarks": lm_list,
                            "contour": None,
                        })
                    hands_list.sort(key=lambda h: h["palm"][0])
                    for idx, h in enumerate(hands_list):
                        h["id"] = idx + 1

            if not hands_list:
                hands_list, debug_mask = detect_multi_hands_from_depth(depth_resized, max_hand_mm)
                if hands_list:
                    tracker_mode = f"TOF-3D ({len(hands_list)} ACTIVE)"

            # 3. Render Skeletons, Gestures, and Pinch-to-Draw Canvas
            active_slots = set()
            for idx, h in enumerate(hands_list):
                active_slots.add(idx)
                _, h_color = HAND_PALETTE[idx % len(HAND_PALETTE)]
                palm_pt = h["palm"]
                draw_pt = h["draw_pt"]
                gesture = h["gesture"]
                dist_m = h["dist_m"]

                if h["landmarks"] is not None:
                    pts = [(int(l.x * vw), int(l.y * vh)) for l in h["landmarks"]]
                    for a, b in HAND_EDGES:
                        cv2.line(output, pts[a], pts[b], h_color, 2, cv2.LINE_AA)
                    for p in pts:
                        cv2.circle(output, p, 3, (255, 255, 255), -1)
                    # Highlight Thumb Tip (#4) and Index Tip (#8)
                    cv2.circle(output, pts[4], 6, (0, 255, 255), -1)
                    cv2.circle(output, pts[8], 6, (0, 255, 255), -1)
                elif h["contour"] is not None:
                    cv2.drawContours(output, [h["contour"]], -1, h_color, 2)

                cv2.circle(output, palm_pt, 7, h_color, -1)
                tag = f"H{h['id']} ({h['side']}): {gesture}"
                if dist_m:
                    tag += f" | {dist_m:.2f}m"
                cv2.putText(output, tag, (max(8, palm_pt[0] - 75), max(24, palm_pt[1] - 25)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, h_color, 1)

                # Pinch-to-Draw
                if draw_enabled and draw_pt is not None:
                    if idx not in smooth_pts:
                        smooth_pts[idx] = np.array(draw_pt, dtype=np.float32)
                    else:
                        smooth_pts[idx] = 0.65 * np.array(draw_pt, dtype=np.float32) + 0.35 * smooth_pts[idx]
                    cur_pt = (int(smooth_pts[idx][0]), int(smooth_pts[idx][1]))

                    if h["is_drawing"]:
                        cv2.circle(output, cur_pt, 9, h_color, -1)
                        cv2.circle(output, cur_pt, 13, (255, 255, 255), 2)
                        if idx in prev_draw_pts and prev_draw_pts[idx] is not None:
                            cv2.line(canvas_layer, prev_draw_pts[idx], cur_pt, h_color, 5, cv2.LINE_AA)
                        prev_draw_pts[idx] = cur_pt
                    elif gesture == "Fist [ERASE]":
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

            # 4. Two-Hand Game Controller Mode
            if game_enabled and HAS_PYAUTOGUI and hands_list:
                now = time.time()
                if (now - last_game_action) > 0.35:
                    for h in hands_list:
                        is_left = (h["side"] == "Left") or (len(hands_list) > 1 and h["id"] == 1)
                        g = h["gesture"]
                        if is_left:
                            if g == "Fist [ERASE]":
                                pyautogui.press("down")
                                last_game_action = now
                            elif g == "Yes / Like":
                                pyautogui.press("left")
                                last_game_action = now
                            elif g == "Peace / Victory":
                                pyautogui.press("right")
                                last_game_action = now
                        else:
                            if g == "Hi / Hello":
                                pyautogui.press("space")
                                last_game_action = now
                            elif g == "Pinch [DRAWING]":
                                pyautogui.click()
                                last_game_action = now

            # 5. Telemetry Dashboard
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            pw = 365
            ui = np.zeros((vh, vw + pw, 3), dtype=np.uint8)
            ui[:, :vw] = output
            panel = ui[:, vw:]
            panel[:] = (22, 22, 28)
            cv2.line(ui, (vw, 0), (vw, vh), (55, 55, 70), 2)

            cv2.putText(panel, "A075V EXPRESSIVE STUDIO", (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 220, 255), 1)
            cv2.putText(panel, f"{tracker_mode} | MAX: {max_hand_mm/1000:.1f}m", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 235, 140), 1)
            cv2.line(panel, (16, 58), (pw - 16, 58), (55, 55, 65), 1)

            y_cursor = 82
            if not hands_list:
                cv2.putText(panel, "Searching for 1-4 hands...", (16, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 135), 1)
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

            emo_str = f"{cached_emo} ({int((cached_conf or 0)*100)}%)" if cached_emo else "Searching Face..."
            cv2.putText(panel, f"EMOTION: {emo_str}", (16, 232), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 120, 255), 1)

            cv2.putText(panel, f"PINCH-DRAW[d]: {'ON' if draw_enabled else 'OFF'}", (16, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 240, 255) if draw_enabled else (110, 110, 110), 1)
            cv2.putText(panel, f"2H-GAME[g]: {'ON' if game_enabled else 'OFF'}", (195, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 255, 120) if game_enabled else (110, 110, 110), 1)

            tw, th, ty = 155, 102, 290
            cv2.putText(panel, "TOF DEPTH MAP", (16, 283), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
            if raw_depth is not None:
                dn = np.clip(raw_depth, 100, 2000) if raw_depth.dtype == np.uint16 else raw_depth
                if raw_depth.dtype == np.uint16:
                    dn = ((dn - 100) / 1900.0 * 255).astype(np.uint8)
                panel[ty : ty + th, 16 : 16 + tw] = cv2.resize(cv2.applyColorMap(dn, cv2.COLORMAP_JET), (tw, th))

            cv2.putText(panel, "PINCH CANVAS / MASK", (188, 283), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
            right_thumb = canvas_layer if np.any(canvas_layer > 0) else cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
            panel[ty : ty + th, 188 : 188 + tw] = cv2.resize(right_thumb, (tw, th))

            cv2.putText(panel, f"FPS: {fps:.1f} | [c]Clear [ [ / ] ]Range [m]Flip [q]Quit",
                        (14, 458), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (165, 165, 180), 1)

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
            elif k == ord("]"):
                max_hand_mm = min(2000, max_hand_mm + 100)
            elif k == ord("["):
                max_hand_mm = max(500, max_hand_mm - 100)

    finally:
        if landmarker is not None:
            landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
