#!/usr/bin/env python3
"""
Hand Gesture + Distance Detector for Sipeed MaixSense-A075V
Features:
  - 3D ToF Depth Foreground Isolation: cuts off walls, curtains, and furniture
  - Unified Dashboard Window (Live Video on Left, Telemetry on Right)
  - Distance-Adaptive scaling with [ and ] range adjustment keys
  - OpenCV 5 / Python 3.14 safe convexity defect handling

Usage: python3 hand_gesture_depth.py
"""

import cv2
import numpy as np
import requests
import struct
import time

# ==================== CONFIGURATION ====================
CAMERA_HOST = "http://192.168.233.1"
RGB_URL = f"{CAMERA_HOST}/rgb"
DEPTH_URL = f"{CAMERA_HOST}/depth"

DEPTH_IS_RAW_UINT16 = True   # True = raw uint16 mm, False = 8-bit depth
DEPTH_MAX_METERS = 2.0       # A075V max range
DEPTH_SAMPLE_RADIUS = 5      # Radius of pixels around center to sample depth

MIN_HAND_MM = 100            # Minimum distance: 10 cm (0.10 m)
DEFAULT_MAX_HAND_MM = 1600   # Max hand detection distance: 1.6 m (adjustable via [ and ])
HAND_SLICE_MM = 130          # Depth slice thickness (13 cm) to isolate hand from body/wall

# Skin color HSV range (fallback)
LOWER_SKIN = np.array([0, 20, 60], dtype=np.uint8)
UPPER_SKIN = np.array([35, 255, 255], dtype=np.uint8)

# ==================== HELPERS: CAMERA ====================

HTTP_SESSION = requests.Session()

def init_camera(host=CAMERA_HOST):
    """Send startup configuration to A075V."""
    try:
        cfg = struct.pack("<BBBBBBBBi", 1, 0, 255, 0, 2, 7, 1, 0, 0)
        HTTP_SESSION.post(f"{host}/set_cfg", data=cfg, timeout=2)
    except Exception:
        pass


def fetch_a075v_frames(host=CAMERA_HOST):
    """Fetch synchronized depth and RGB frame from A075V getdeep endpoint."""
    try:
        resp = HTTP_SESSION.get(f"{host}/getdeep", timeout=1.5)
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
        resp = HTTP_SESSION.get(url, timeout=2)
        resp.raise_for_status()
        if len(resp.content) == 0:
            _, rgb = fetch_a075v_frames()
            return rgb
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def fetch_depth(url):
    try:
        resp = HTTP_SESSION.get(url, timeout=2)
        resp.raise_for_status()
        if len(resp.content) == 0:
            depth, _ = fetch_a075v_frames()
            return depth
        arr = np.frombuffer(resp.content, dtype=np.uint16 if DEPTH_IS_RAW_UINT16 else np.uint8)
        return arr.reshape((240, 320))
    except Exception:
        return None


# ==================== COMPUTER VISION PIPELINE ====================

def segment_hand(frame, depth_resized=None, max_dist_mm=DEFAULT_MAX_HAND_MM):
    """
    3D ToF Depth Foreground Isolation:
    Isolates ONLY the closest foreground object (within max_dist_mm),
    completely blacking out walls, bed, and curtains.
    """
    if depth_resized is not None and depth_resized.dtype == np.uint16:
        valid_hand_pixels = depth_resized[
            (depth_resized > MIN_HAND_MM) & (depth_resized < max_dist_mm)
        ]
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

    # Fallback to HSV skin mask if depth is unavailable
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_hand(mask):
    """Find hand contour above distance-adaptive area threshold."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    # Rejects noise (<550 px) or full-screen blobs
    if area < 550 or area > (mask.shape[0] * mask.shape[1] * 0.55):
        return None
    return largest


def count_finger_valleys(hand_contour, draw_frame=None):
    """Count fingertips via convexity defects (OpenCV 5 / Python 3.14 safe)."""
    hull = cv2.convexHull(hand_contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return 0, []

    try:
        defects = cv2.convexityDefects(hand_contour, hull)
    except cv2.error:
        return 0, []

    if defects is None:
        return 0, []

    # Safe reshape to (-1, 4) prevents TypeError on 2D defect arrays
    defects_2d = defects.reshape(-1, 4)
    _, _, bw, bh = cv2.boundingRect(hand_contour)
    min_defect_depth = max(7.0, bh * 0.12)

    valleys = []
    for s, e, f, d in defects_2d:
        start = tuple(hand_contour[int(s)][0])
        end = tuple(hand_contour[int(e)][0])
        far = tuple(hand_contour[int(f)][0])

        a = np.linalg.norm(np.array(start) - np.array(end))
        b = np.linalg.norm(np.array(start) - np.array(far))
        c = np.linalg.norm(np.array(end) - np.array(far))

        angle = np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c + 1e-6), -1.0, 1.0))
        depth_defect = d / 256.0

        if angle < np.deg2rad(85) and depth_defect > min_defect_depth:
            valleys.append(far)
            if draw_frame is not None:
                cv2.circle(draw_frame, start, 5, (0, 165, 255), -1)  # Fingertip

    fingers = min(len(valleys) + 1, 5) if len(valleys) > 0 else 0
    return fingers, valleys


def classify_gesture(fingers, contour=None):
    """Map finger count to human-readable gesture with Fist vs 1 Finger discrimination."""
    if contour is not None and fingers == 0:
        hull_pts = cv2.convexHull(contour, returnPoints=True)
        solidity = float(cv2.contourArea(contour)) / (cv2.contourArea(hull_pts) + 1e-5)
        if solidity < 0.80:
            return "One finger"
        return "Fist"

    gestures = {
        0: "Fist",
        1: "One finger",
        2: "Peace / V",
        3: "Three fingers",
        4: "Four fingers",
        5: "Open palm",
    }
    return gestures.get(fingers, "Open palm")


def get_distance_at_point(depth_img, point, r=DEPTH_SAMPLE_RADIUS):
    """Sample median depth around hand center to eliminate noise. Returns meters."""
    if depth_img is None:
        return None

    cx, cy = int(point[0]), int(point[1])
    h, w = depth_img.shape[:2]
    patch = depth_img[max(0, cy - r):min(h, cy + r + 1),
                      max(0, cx - r):min(w, cx + r + 1)]
    if patch.size == 0:
        return None

    if depth_img.dtype == np.uint16 or DEPTH_IS_RAW_UINT16:
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 1000.0
    else:
        return float(np.median(patch)) / 255.0 * DEPTH_MAX_METERS


def colorize_depth(depth_map):
    """Convert depth map into a jet heatmap for visualization."""
    if depth_map is None:
        return np.zeros((160, 213, 3), dtype=np.uint8)
    if depth_map.dtype == np.uint16:
        norm = np.clip(depth_map, 100, 2000)
        norm = ((norm - 100) / 1900.0 * 255).astype(np.uint8)
    else:
        norm = depth_map
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return colored


# ==================== SIDE PANEL DASHBOARD UI ====================

def build_combined_frame(video_frame, depth_map, mask, detected, gesture, fingers, distance_m, fps, max_hand_mm):
    """
    Renders unified window:
    [ Left: 640x480 Live Video ] | [ Right: 320x480 Dashboard ]
    """
    vh, vw = video_frame.shape[:2]
    panel_w = 320
    canvas = np.zeros((vh, vw + panel_w, 3), dtype=np.uint8)

    # 1. Video on Left
    canvas[:, :vw] = video_frame

    # 2. Side Panel Background
    panel = canvas[:, vw:]
    panel[:] = (24, 24, 28)

    cv2.line(canvas, (vw, 0), (vw, vh), (50, 50, 60), 2)

    # 3. Header
    cv2.putText(panel, "MAIXSENSE-A075V", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1)
    cv2.putText(panel, "3D TOF DEPTH-ISOLATED", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 230, 140), 1)
    cv2.line(panel, (20, 62), (panel_w - 20, 62), (60, 60, 70), 1)

    # 4. Status Indicator Badge
    y_stat = 90
    if detected:
        cv2.circle(panel, (30, y_stat), 8, (0, 220, 100), -1)
        cv2.putText(panel, "HAND DETECTED", (48, y_stat + 5), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 120), 1)
    else:
        cv2.circle(panel, (30, y_stat), 8, (80, 80, 220), -1)
        cv2.putText(panel, "SEARCHING...", (48, y_stat + 5), cv2.FONT_HERSHEY_DUPLEX, 0.55, (100, 100, 220), 1)

    # 5. Gesture Info Block
    cv2.putText(panel, "GESTURE:", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 170), 1)
    g_color = (0, 255, 255) if detected else (100, 100, 110)
    cv2.putText(panel, gesture, (20, 158), cv2.FONT_HERSHEY_DUPLEX, 0.75, g_color, 2)

    f_text = f"Fingers Count: {fingers}" if detected else "Fingers Count: 0"
    cv2.putText(panel, f_text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 210), 1)

    # 6. Distance Info Block
    cv2.putText(panel, "METRIC DISTANCE:", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 170), 1)
    if distance_m is not None:
        d_color = (50, 220, 255)
        d_val = f"{distance_m:.2f} m"
        d_sub = f"({int(distance_m * 1000)} mm)"
    else:
        d_color = (100, 100, 110)
        d_val = "-- m"
        d_sub = "(-- mm)"

    cv2.putText(panel, d_val, (20, 245), cv2.FONT_HERSHEY_DUPLEX, 0.85, d_color, 2)
    cv2.putText(panel, d_sub, (130, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 150), 1)

    cv2.line(panel, (20, 260), (panel_w - 20, 260), (60, 60, 70), 1)

    # 7. Thumbnails (Depth map + Clean Depth Mask)
    tw, th = 130, 95
    thumb_y = 295

    cv2.putText(panel, "DEPTH MAP", (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    color_depth = colorize_depth(depth_map)
    depth_small = cv2.resize(color_depth, (tw, th))
    panel[thumb_y:thumb_y + th, 20:20 + tw] = depth_small
    cv2.rectangle(panel, (20, thumb_y), (20 + tw, thumb_y + th), (60, 60, 70), 1)

    cv2.putText(panel, f"3D MASK ({max_hand_mm/1000:.1f}m)", (170, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
    mask_bgr = cv2.cvtColor(cv2.resize(mask, (tw, th)), cv2.COLOR_GRAY2BGR)
    panel[thumb_y:thumb_y + th, 170:170 + tw] = mask_bgr
    cv2.rectangle(panel, (170, thumb_y), (170 + tw, thumb_y + th), (60, 60, 70), 1)

    # 8. Footer Telemetry
    cv2.line(panel, (20, 420), (panel_w - 20, 420), (60, 60, 70), 1)
    cv2.putText(panel, f"FPS: {fps:.1f} | Range: [ {max_hand_mm/1000:.2f}m ]", (20, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 190), 1)
    cv2.putText(panel, "Press '[' / ']' for Range | 'q' to exit", (20, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 140), 1)

    return canvas


# ==================== MAIN LOOP ====================

def main():
    print("=====================================================")
    print(" Starting MaixSense-A075V Hand Gesture + Distance")
    print(" Target IP :", CAMERA_HOST)
    print(" Features  : 3D ToF Depth Foreground Isolation (Zero Background Bleed)")
    print(" Hotkeys   : '[' Lower Range  |  ']' Higher Range  |  'q' Quit")
    print("=====================================================\n")

    init_camera()
    prev_time = time.time()
    max_hand_mm = DEFAULT_MAX_HAND_MM

    window_name = "MaixSense-A075V - Gesture & Distance Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        depth, rgb = fetch_a075v_frames()
        if rgb is None:
            rgb = fetch_rgb(RGB_URL)
            depth = fetch_depth(DEPTH_URL)

        if rgb is None:
            time.sleep(0.05)
            continue

        raw_depth = depth

        # Resize depth to RGB resolution
        if depth is not None and depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        output_video = rgb.copy()

        # 1. 3D Depth Isolation: cuts off walls, curtains, and body!
        mask = segment_hand(rgb, depth, max_dist_mm=max_hand_mm)
        hand = find_hand(mask)

        detected = (hand is not None)
        gesture = "No Hand"
        fingers = 0
        distance_m = None

        if detected:
            epsilon = 0.004 * cv2.arcLength(hand, True)
            hand_smooth = cv2.approxPolyDP(hand, epsilon, True)
            cv2.drawContours(output_video, [hand_smooth], -1, (0, 255, 100), 2)

            M = cv2.moments(hand)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(output_video, (cx, cy), 8, (255, 100, 0), -1)
                cv2.circle(output_video, (cx, cy), 12, (255, 255, 255), 1)

                distance_m = get_distance_at_point(depth, (cx, cy))

            fingers, valleys = count_finger_valleys(hand, draw_frame=output_video)
            gesture = classify_gesture(fingers, contour=hand)

            for pt in valleys:
                cv2.circle(output_video, pt, 6, (0, 255, 255), -1)

            cv2.putText(output_video, f"{gesture} | {distance_m:.2f}m" if distance_m else gesture,
                        (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 100), 2)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
        prev_time = curr_time

        combined = build_combined_frame(
            output_video, raw_depth, mask, detected, gesture, fingers, distance_m, fps, max_hand_mm
        )

        cv2.imshow(window_name, combined)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord(']'):
            max_hand_mm = min(2000, max_hand_mm + 100)
            print(f"[A075V] Max Hand Distance increased to: {max_hand_mm / 1000:.2f} m")
        elif k == ord('['):
            max_hand_mm = max(400, max_hand_mm - 100)
            print(f"[A075V] Max Hand Distance decreased to: {max_hand_mm / 1000:.2f} m")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
