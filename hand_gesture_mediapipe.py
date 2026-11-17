#!/usr/bin/env python3
"""
Hand Gesture + Distance Detector using MediaPipe + Sipeed MaixSense-A075V Depth
Requires: opencv-python, numpy, requests, mediapipe
Optional (Hugging Face): transformers, torch, pillow
Usage: python3 hand_gesture_mediapipe.py

Notes:
- MediaPipe Hands detects 21 3D landmarks per hand in real time.
- Landmark (x, y) is mapped to the A075V depth map to get METRIC distance (meters).
- Includes a lightweight rule-based gesture classifier (no heavy HF model required).
- Optional: uncomment the HF section to load a gesture classifier from Hugging Face.
"""

import cv2
import numpy as np
import requests
import struct
import time

# ==================== MEDIAPIPE ====================
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ==================== CONFIGURATION ====================
RGB_URL = "http://192.168.233.1/rgb"
DEPTH_URL = "http://192.168.233.1/depth"

DEPTH_IS_RAW_UINT16 = True   # True = raw uint16 mm, False = 8-bit visualized depth
DEPTH_MAX_METERS = 2.0       # A075V max range
DEPTH_SAMPLE_RADIUS = 4      # pixels around landmark to average for depth

# ==================== HELPERS: CAMERA FETCH ====================

def init_camera(host="http://192.168.233.1"):
    try:
        cfg = struct.pack("<BBBBBBBBi", 1, 0, 255, 0, 2, 7, 1, 0, 0)
        requests.post(f"{host}/set_cfg", data=cfg, timeout=3)
    except Exception:
        pass


def fetch_a075v_frames(host="http://192.168.233.1"):
    try:
        resp = requests.get(f"{host}/getdeep", timeout=3)
        data = resp.content
        if len(data) >= 36:
            _, deep_mode, _, ir_mode, status_mode, _, rgb_mode, _, _ = struct.unpack('<BBBBBBBBi', data[16:28])
            deep_data_size, rgb_data_size = struct.unpack('<ii', data[28:36])
            payload = data[36:]

            deepth_size = (320 * 240 * 2) >> deep_mode
            ir_size = (320 * 240 * 2) >> ir_mode
            status_size = (320 * 240 // 8) * (16 if status_mode == 0 else 2 if status_mode == 1 else 8 if status_mode == 2 else 1)

            depth_bytes = payload[:deepth_size]
            dtype = np.uint16 if deep_mode == 0 else np.uint8
            depth = np.frombuffer(depth_bytes, dtype=dtype).reshape((240, 320)) if deepth_size > 0 else None

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


def fetch_rgb(url):
    try:
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        if len(resp.content) == 0:
            _, rgb = fetch_a075v_frames()
            return rgb
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[RGB] Error: {e}")
        return None


def fetch_depth(url):
    try:
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        if len(resp.content) == 0:
            depth, _ = fetch_a075v_frames()
            return depth
        if DEPTH_IS_RAW_UINT16:
            arr = np.frombuffer(resp.content, dtype=np.uint16)
            depth = arr.reshape((240, 320))
        else:
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if depth is None:
                return None
            if len(depth.shape) == 3:
                depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        return depth
    except Exception as e:
        print(f"[DEPTH] Error: {e}")
        return None


def resize_depth_to_rgb(depth, rgb_shape):
    if depth is None:
        return None
    h, w = rgb_shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth


def get_distance_at_point(depth_img, point, scale=1.0):
    if depth_img is None:
        return None
    x, y = int(point[0]), int(point[1])
    h, w = depth_img.shape[:2]
    r = DEPTH_SAMPLE_RADIUS
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    patch = depth_img[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    if DEPTH_IS_RAW_UINT16:
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 1000.0 * scale
    else:
        return float(np.median(patch)) / 255.0 * DEPTH_MAX_METERS * scale


# ==================== HELPERS: GESTURE CLASSIFICATION ====================

# MediaPipe hand landmark indices
# 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]


def finger_states(landmarks, handedness_label):
    """Determine which fingers are up using landmark geometry."""
    lm = landmarks
    states = [False] * 5

    thumb_tip = lm[FINGER_TIPS[0]]
    thumb_ip = lm[FINGER_PIPS[0]]
    if handedness_label == "Right":
        states[0] = thumb_tip.x < thumb_ip.x
    else:
        states[0] = thumb_tip.x > thumb_ip.x

    for i in range(1, 5):
        tip = lm[FINGER_TIPS[i]]
        pip = lm[FINGER_PIPS[i]]
        states[i] = tip.y < pip.y

    return states


def classify_gesture(states):
    """Map finger states to a gesture name."""
    thumb, index, middle, ring, pinky = states
    count = sum(states)

    if count == 0:
        return "Fist"
    if count == 5:
        return "Open Palm"
    if states == [False, True, False, False, False]:
        return "Index / 1"
    if states == [False, True, True, False, False]:
        return "Peace / V"
    if states == [False, True, True, True, False]:
        return "Three / W"
    if states == [False, True, True, True, True]:
        return "Four"
    if states == [True, False, False, False, False]:
        return "Thumb Up"
    if states == [True, True, False, False, True]:
        return "OK / Pinch"
    if states == [True, False, False, False, True]:
        return "Call Me"
    if count == 1:
        return "One Finger"
    if count == 2:
        return "Two Fingers"
    if count == 3:
        return "Three Fingers"
    if count == 4:
        return "Four Fingers"
    return "Unknown"


def handedness_from_results(multi_handedness, hand_index):
    if multi_handedness and hand_index < len(multi_handedness):
        return multi_handedness[hand_index].classification[0].label
    return "Unknown"


# ==================== OPTIONAL: HUGGING FACE GESTURE CLASSIFIER ====================
# Uncomment the block below if you want to use a HF image model instead of (or with)
# the rule-based classifier. Requires: transformers, torch, pillow.
#
# from transformers import pipeline
# from PIL import Image
#
# HF_GESTURE_MODEL = "microsoft/resnet-50"  # placeholder; swap for a gesture model
# hf_classifier = None
#
# def load_hf_classifier():
#     global hf_classifier
#     if hf_classifier is None:
#         print("[HF] Loading gesture model...")
#         hf_classifier = pipeline("image-classification", model=HF_GESTURE_MODEL)
#
# def hf_gesture(frame_rgb):
#     if hf_classifier is None:
#         return None
#     pil_img = Image.fromarray(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
#     results = hf_classifier(pil_img)
#     return results[0]["label"], results[0]["score"]


# ==================== MAIN ====================

def main():
    print("Starting MediaPipe Hand Gesture + Distance detection...")
    print("Camera RGB :", RGB_URL)
    print("Camera Depth:", DEPTH_URL)
    print("Press 'q' to quit.\n")

    init_camera()
    prev_time = 0

    while True:
        depth, rgb = fetch_a075v_frames()
        if rgb is None:
            rgb = fetch_rgb(RGB_URL)
            depth = fetch_depth(DEPTH_URL)

        if rgb is None:
            print("Could not fetch RGB frame. Retrying...")
            time.sleep(0.5)
            continue

        depth = resize_depth_to_rgb(depth, rgb.shape)

        rgb_for_mp = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_for_mp)

        output = rgb.copy()
        h, w = output.shape[:2]

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    output,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                handedness = handedness_from_results(results.multi_handedness, idx)
                states = finger_states(hand_landmarks.landmark, handedness)
                gesture = classify_gesture(states)

                ref_lm = hand_landmarks.landmark[8]
                cx = int(ref_lm.x * w)
                cy = int(ref_lm.y * h)

                distance_m = get_distance_at_point(depth, (cx, cy))

                cv2.circle(output, (cx, cy), 8, (0, 255, 255), -1)
                cv2.putText(output, f"{handedness}: {gesture}", (10, 30 + idx * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                dist_text = f"{distance_m:.2f} m" if distance_m is not None else "N/A"
                cv2.putText(output, f"Distance: {dist_text}", (10, 60 + idx * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(output, f"FPS: {fps:.1f}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("MediaPipe Hand Gesture + Distance", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

