#!/usr/bin/env python3
"""
MaixSense-A075V All-In-One Vision Studio (Zero-OOM, Distance-Adaptive & Depth-Isolated)
- Zero-Background Bleed: Hardware ToF Depth gating cuts off walls, bed, and curtains
- Distance-Adaptive: Operates up to 1.8m - 2.0m with live `[` and `]` range tuning hotkeys
- Low-RAM Architecture: Native cv2.dnn for Emotion + Lite MediaPipe (Zero OOM / "Killed" crash)
- Features: 3D Hand Gesture + Air Drawing + Real-Life Emotion + Hand & Body Distance + Games
"""

import os
# Limit C++ thread pools BEFORE importing cv2/mediapipe to prevent Linux OOM killer
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"

import time
import struct
import cv2
import numpy as np
import requests

cv2.setNumThreads(2)

# Optional PyAutoGUI for Game Mode
try:
    import pyautogui
    pyautogui.PAUSE = 0.0
    pyautogui.FAILSAFE = True
    HAS_PYAUTOGUI = True
except Exception:
    HAS_PYAUTOGUI = False

# Optional MediaPipe (Lite model_complexity=0 to save RAM)
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    HAS_MEDIAPIPE = True
except Exception:
    pass

CAMERA_HOST = "http://192.168.233.1"
HTTP_SESSION = requests.Session()

# ==================== DISTANCE CONFIGURATION ====================
MIN_HAND_MM = 100        # Minimum distance: 10 cm (0.10 m)
DEFAULT_MAX_HAND_MM = 1600 # Default max distance: 1.6 m (adjustable with '[' and ']')
HAND_SLICE_MM = 130      # Depth thickness of hand/wrist (13 cm) before cutting off body/wall


# ==================== LIGHTWEIGHT OPENCV-NATIVE EMOTION DETECTOR ====================

class LightweightEmotionDetector:
    """Uses native cv2.dnn instead of onnxruntime so it never triggers OOM 'Killed'."""
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
            except Exception as e:
                print(f"[Face] YuNet init warning: {e}")

        if os.path.exists(emo_path):
            try:
                self.emo_net = cv2.dnn.readNetFromONNX(emo_path)
            except Exception as e:
                print(f"[Emotion] cv2.dnn ONNX warning: {e}")

    def detect(self, frame):
        if self.face_det is None:
            return None, None, None
        h, w = frame.shape[:2]
        self.face_det.setInputSize((w, h))
        _, faces = self.face_det.detect(frame)
        if faces is None or len(faces) == 0:
            return None, None, None

        # Pick largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        x, y = max(0, x), max(0, y)
        fw, fh = min(fw, w - x), min(fh, h - y)
        if fw < 35 or fh < 35:
            return None, None, None

        if self.emo_net is None:
            return "Face", 1.0, (x, y, fw, fh)

        try:
            roi = frame[y : y + fh, x : x + fw]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            blob = resized.reshape(1, 1, 64, 64)
            self.emo_net.setInput(blob)
            logits = self.emo_net.forward()[0]
            logits = logits[:7]
            exp_scores = np.exp(logits - np.max(logits))
            probs = exp_scores / np.sum(exp_scores)
            idx = int(np.argmax(probs))
            return self.EMOTIONS[idx], float(probs[idx]), (x, y, fw, fh)
        except Exception:
            return "Face", 1.0, (x, y, fw, fh)


# ==================== CAMERA FETCH ====================

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
        if len(data) >= 36:
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
            if rgb is not None:
                return depth, rgb
    except Exception:
        pass
    return None, None


# ==================== 3D TOF DEPTH ISOLATION & HAND TRACKING ====================

def build_clean_hand_mask(rgb_frame, depth_resized, max_dist_mm=DEFAULT_MAX_HAND_MM):
    """
    Isolates hand up to max_dist_mm (e.g. 1.8m) while slicing off body, walls, and curtains.
    """
    if depth_resized is not None and depth_resized.dtype == np.uint16:
        valid_hand_pixels = depth_resized[
            (depth_resized > MIN_HAND_MM) & (depth_resized < max_dist_mm)
        ]
        # Low threshold (180 pixels) supports far hands at 1.5m - 1.8m
        if valid_hand_pixels.size > 180:
            closest_mm = float(np.percentile(valid_hand_pixels, 3))
            depth_mask = (
                (depth_resized >= max(80.0, closest_mm - 30.0)) &
                (depth_resized <= closest_mm + HAND_SLICE_MM)
            ).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
            depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
            return depth_mask

    # Fallback if depth map is 8-bit
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([0, 25, 60], np.uint8), np.array([35, 255, 255], np.uint8))


def analyze_depth_hand_contour(mask, draw_frame):
    """Distance-adaptive contour analyzer with safe defect unpacking."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "No Hand", 0, None, None

    hand = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(hand)

    # Adaptive minimum area: 550 allows hands up to 1.8m away
    if area < 550 or area > (mask.shape[0] * mask.shape[1] * 0.55):
        return False, "No Hand", 0, None, None

    epsilon = 0.004 * cv2.arcLength(hand, True)
    hand = cv2.approxPolyDP(hand, epsilon, True)
    cv2.drawContours(draw_frame, [hand], -1, (0, 255, 100), 2)

    M = cv2.moments(hand)
    if M["m00"] == 0:
        return False, "No Hand", 0, None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    topmost = tuple(hand[hand[:, :, 1].argmin()][0])

    _, _, bw, bh = cv2.boundingRect(hand)
    min_defect_depth = max(7.0, bh * 0.12)

    hull_idx = cv2.convexHull(hand, returnPoints=False)
    hull_pts = cv2.convexHull(hand, returnPoints=True)
    solidity = float(area) / (cv2.contourArea(hull_pts) + 1e-5)

    if hull_idx is None or len(hull_idx) < 4:
        return True, "Fist", 0, (cx, cy), topmost

    try:
        defects = cv2.convexityDefects(hand, hull_idx)
    except cv2.error:
        defects = None

    valleys = []
    if defects is not None:
        defects_2d = defects.reshape(-1, 4)
        for s, e, f, d in defects_2d:
            start = tuple(hand[int(s)][0])
            end = tuple(hand[int(e)][0])
            far = tuple(hand[int(f)][0])

            a = np.linalg.norm(np.array(start) - np.array(end))
            b = np.linalg.norm(np.array(start) - np.array(far))
            c = np.linalg.norm(np.array(end) - np.array(far))
            angle = np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c + 1e-6), -1.0, 1.0))
            depth_defect = d / 256.0

            if angle < np.deg2rad(85) and depth_defect > min_defect_depth and far[1] < (cy + bh * 0.35):
                valleys.append(far)
                cv2.circle(draw_frame, far, 5, (0, 255, 255), -1)
                cv2.circle(draw_frame, start, 6, (0, 165, 255), -1)

    if len(valleys) == 0:
        if solidity > 0.82:
            return True, "Fist", 0, (cx, cy), topmost
        cv2.circle(draw_frame, topmost, 8, (0, 255, 255), -1)
        return True, "Index / Draw", 1, (cx, cy), topmost

    fingers = min(len(valleys) + 1, 5)
    g_map = {
        1: "Index / Draw",
        2: "Peace / Hover",
        3: "Three Fingers",
        4: "Four Fingers",
        5: "Open Palm",
    }
    return True, g_map.get(fingers, "Open Palm"), fingers, (cx, cy), topmost


def classify_mediapipe(hl, vw, vh):
    lm = hl.landmark
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    states = [False] * 5

    pinky_mcp = np.array([lm[17].x, lm[17].y])
    thumb_tip = np.array([lm[4].x, lm[4].y])
    thumb_ip = np.array([lm[3].x, lm[3].y])
    states[0] = np.linalg.norm(thumb_tip - pinky_mcp) > np.linalg.norm(thumb_ip - pinky_mcp) * 1.15

    for i in range(1, 5):
        states[i] = lm[tips[i]].y < lm[pips[i]].y

    count = sum(states)
    pinch = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)

    if pinch < 0.045 and states[2] and states[3]:
        gesture = "OK / Pinch"
    elif count == 0:
        gesture = "Fist"
    elif states[1] and not states[2] and not states[3] and not states[4]:
        gesture = "Index / Draw"
        count = 1
    elif states[1] and states[2] and not states[3] and not states[4]:
        gesture = "Peace / Hover"
        count = 2
    elif states == [True, False, False, False, False]:
        gesture = "Thumb Up"
    elif count == 3:
        gesture = "Three Fingers"
    elif count == 4:
        gesture = "Four Fingers"
    elif count == 5:
        gesture = "Open Palm"
    else:
        gesture = f"{count} Fingers"

    cx = int(((lm[0].x + lm[5].x + lm[17].x) / 3.0) * vw)
    cy = int(((lm[0].y + lm[5].y + lm[17].y) / 3.0) * vh)
    ix, iy = int(lm[8].x * vw), int(lm[8].y * vh)
    return gesture, count, (cx, cy), (ix, iy)


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


# ==================== MAIN LOOP ====================

def main():
    print("=================================================================")
    print(" MaixSense-A075V AI Vision Studio (Zero-OOM & Depth-Isolated)")
    print(" Hotkeys:")
    print("   [d] Toggle Air Drawing          [c] Clear Canvas")
    print("   [k] Cycle Brush Color           [g] Toggle Game Controller")
    print("   [[] Lower Max Hand Range        []] Increase Max Hand Range")
    print("   [m] Mirror/Flip Camera          [q] Quit")
    print("=================================================================")

    init_camera()
    emo_detector = LightweightEmotionDetector()

    mp_detector = None
    if HAS_MEDIAPIPE:
        mp_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    draw_enabled = True
    game_enabled = False
    mirror = True
    max_hand_mm = DEFAULT_MAX_HAND_MM

    colors = [
        ("Cyan", (255, 240, 0)),
        ("Green", (50, 255, 50)),
        ("Magenta", (255, 50, 255)),
        ("Gold", (0, 215, 255)),
    ]
    color_idx = 0
    canvas_layer = None
    smooth_pt = None
    prev_pt = None

    last_game_time = 0
    last_game_gesture = "No Hand"
    frame_count = 0
    cached_emo, cached_conf, cached_face = None, None, None
    prev_time = time.time()

    win_name = "MaixSense-A075V - AI Vision Studio"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    while True:
        depth, rgb = fetch_a075v_frames()
        if rgb is None:
            time.sleep(0.03)
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

        output = rgb.copy()

        # 1. Build 3D ToF Depth-Isolated Hand Mask (Eliminates wall, bed, and curtains!)
        clean_mask = build_clean_hand_mask(rgb, depth_resized, max_dist_mm=max_hand_mm)

        # 2. Emotion & Face/Body Distance (Every 4th frame)
        frame_count += 1
        if frame_count % 4 == 0:
            cached_emo, cached_conf, cached_face = emo_detector.detect(rgb)

        body_dist_m = None
        if cached_face is not None:
            fx, fy, fw, fh = cached_face
            body_dist_m = sample_distance_m(depth_resized, fx + fw // 2, fy + fh // 2, r=10)
            cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh), (255, 50, 255), 2)
            lbl = f"{cached_emo} ({int((cached_conf or 0)*100)}%)"
            if body_dist_m:
                lbl += f" | {body_dist_m:.2f}m"
            cv2.putText(output, lbl, (fx, max(20, fy - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 100, 255), 1)
        else:
            body_dist_m = sample_distance_m(depth_resized, vw // 2, vh // 2, r=15)

        # 3. Hand Detection
        detected = False
        gesture = "No Hand"
        fingers = 0
        palm_pt = None
        index_pt = None
        hand_dist_m = None
        engine_name = "TOF-DEPTH 3D"

        if mp_detector is not None:
            res = mp_detector.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(output, hl, mp_hands.HAND_CONNECTIONS)
                gesture, fingers, palm_pt, index_pt = classify_mediapipe(hl, vw, vh)
                detected = True
                engine_name = "MEDIAPIPE 3D"

        if not detected:
            detected, gesture, fingers, palm_pt, index_pt = analyze_depth_hand_contour(clean_mask, output)

        if detected and palm_pt is not None:
            hand_dist_m = sample_distance_m(depth_resized, palm_pt[0], palm_pt[1], r=6)
            cv2.circle(output, palm_pt, 7, (255, 100, 0), -1)
            cv2.circle(output, palm_pt, 11, (255, 255, 255), 1)
            ov_txt = f"{gesture} | {hand_dist_m:.2f}m" if hand_dist_m else gesture
            cv2.putText(output, ov_txt, (15, 34), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 100), 2)

        # 4. Interactive Screen Drawing
        if draw_enabled and detected and index_pt is not None:
            if smooth_pt is None:
                smooth_pt = np.array(index_pt, dtype=np.float32)
            else:
                smooth_pt = 0.65 * np.array(index_pt, dtype=np.float32) + 0.35 * smooth_pt
            cur_pt = (int(smooth_pt[0]), int(smooth_pt[1]))
            _, b_col = colors[color_idx]

            if gesture == "Index / Draw":
                cv2.circle(output, cur_pt, 9, b_col, -1)
                if prev_pt is not None:
                    cv2.line(canvas_layer, prev_pt, cur_pt, b_col, 5, cv2.LINE_AA)
                prev_pt = cur_pt
            elif gesture == "Fist" and palm_pt is not None:
                cv2.circle(output, palm_pt, 28, (0, 0, 255), 2)
                cv2.circle(canvas_layer, palm_pt, 28, (0, 0, 0), -1)
                prev_pt = None
            else:
                cv2.circle(output, cur_pt, 7, (220, 220, 220), 2)
                prev_pt = None
        else:
            prev_pt = None
            smooth_pt = None

        # Blend Air Canvas
        c_gray = cv2.cvtColor(canvas_layer, cv2.COLOR_BGR2GRAY)
        _, c_mask = cv2.threshold(c_gray, 5, 255, cv2.THRESH_BINARY)
        output[c_mask > 0] = cv2.addWeighted(output, 0.25, canvas_layer, 0.75, 0)[c_mask > 0]

        # 5. Game Controller Actions
        if game_enabled and HAS_PYAUTOGUI and detected:
            now = time.time()
            if gesture != last_game_gesture and (now - last_game_time) > 0.45:
                if gesture == "Open Palm":
                    pyautogui.press("space")
                elif gesture == "Fist":
                    pyautogui.press("down")
                elif gesture == "Peace / Hover":
                    pyautogui.click()
                last_game_time = now
                last_game_gesture = gesture

        # 6. Dashboard UI
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        pw = 330
        ui = np.zeros((vh, vw + pw, 3), dtype=np.uint8)
        ui[:, :vw] = output
        panel = ui[:, vw:]
        panel[:] = (24, 24, 29)
        cv2.line(ui, (vw, 0), (vw, vh), (55, 55, 70), 2)

        cv2.putText(panel, "MAIXSENSE-A075V STUDIO", (18, 30), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 220, 255), 1)
        cv2.putText(panel, f"TRACKER: {engine_name}", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 230, 140), 1)
        cv2.line(panel, (18, 60), (pw - 18, 60), (55, 55, 65), 1)

        cv2.circle(panel, (28, 84), 7, (0, 240, 110) if detected else (80, 80, 220), -1)
        cv2.putText(panel, f"{gesture} ({fingers}F)", (44, 89), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255) if detected else (120, 120, 130), 1)

        cv2.putText(panel, "HAND DISTANCE:", (18, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 175), 1)
        hd_txt = f"{hand_dist_m:.2f} m ({int(hand_dist_m*1000)} mm)" if hand_dist_m else "-- m"
        cv2.putText(panel, hd_txt, (18, 150), cv2.FONT_HERSHEY_DUPLEX, 0.65, (50, 230, 255), 1)

        cv2.putText(panel, "PERSON / FACE DISTANCE:", (18, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 175), 1)
        bd_txt = f"{body_dist_m:.2f} m ({int(body_dist_m*1000)} mm)" if body_dist_m else "-- m"
        cv2.putText(panel, bd_txt, (18, 210), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 170, 70), 1)

        emo_txt = f"{cached_emo} ({int((cached_conf or 0)*100)}%)" if cached_emo else "Searching Face..."
        cv2.putText(panel, f"EMOTION: {emo_txt}", (18, 245), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 120, 255), 1)

        c_name, c_bgr = colors[color_idx]
        cv2.putText(panel, f"DRAW[d]: {'ON ('+c_name+')' if draw_enabled else 'OFF'}", (18, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c_bgr if draw_enabled else (110, 110, 110), 1)
        cv2.putText(panel, f"GAME[g]: {'ON' if game_enabled else 'OFF'}", (205, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120) if game_enabled else (110, 110, 110), 1)

        # Thumbnails: Depth Map & Clean Depth-Isolated Hand Mask
        tw, th, ty = 135, 98, 305
        cv2.putText(panel, "DEPTH MAP", (18, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
        if raw_depth is not None:
            dn = np.clip(raw_depth, 100, 2000) if raw_depth.dtype == np.uint16 else raw_depth
            if raw_depth.dtype == np.uint16:
                dn = ((dn - 100) / 1900.0 * 255).astype(np.uint8)
            panel[ty : ty + th, 18 : 18 + tw] = cv2.resize(cv2.applyColorMap(dn, cv2.COLORMAP_JET), (tw, th))

        cv2.putText(panel, f"3D MASK ({max_hand_mm/1000:.1f}m [ ])", (172, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
        panel[ty : ty + th, 172 : 172 + tw] = cv2.cvtColor(cv2.resize(clean_mask, (tw, th)), cv2.COLOR_GRAY2BGR)

        cv2.putText(panel, f"FPS: {fps:.1f} | [[]/[]]Range [c]Clear [q]Quit", (18, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 175), 1)

        cv2.imshow(win_name, ui)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("c"):
            canvas_layer[:] = 0
        elif k == ord("d"):
            draw_enabled = not draw_enabled
        elif k == ord("k"):
            color_idx = (color_idx + 1) % len(colors)
        elif k == ord("g"):
            game_enabled = not game_enabled
        elif k == ord("m"):
            mirror = not mirror
        elif k == ord("]"):
            max_hand_mm = min(2000, max_hand_mm + 100)
            print(f"[Studio] Max Hand Range increased to: {max_hand_mm / 1000:.2f} m")
        elif k == ord("["):
            max_hand_mm = max(400, max_hand_mm - 100)
            print(f"[Studio] Max Hand Range decreased to: {max_hand_mm / 1000:.2f} m")

    if mp_detector is not None:
        mp_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
