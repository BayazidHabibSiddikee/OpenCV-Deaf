#!/home/bayazid/.opencv/bin/python3
"""
Hand-as-Mouse Controller
Your hand becomes the mouse cursor. Gestures perform clicks.

Gestures:
  Hand position  -> Move cursor
  Peace (V sign) -> Left click
  Fist           -> Right click
  Thumb Up       -> Scroll up
  Open Palm      -> Scroll down

Run: python3 games/mouse.py
"""

import sys
import time
import cv2
import numpy as np
import requests

sys.path.insert(0, "/home/bayazid/Documents/OpenCV/sign_app")
from camera import CameraProcessor
from gesture_controller import GestureController

RGB_URL = "http://192.168.233.1/rgb"
DEPTH_URL = "http://192.168.233.1/depth"


def main():
    camera = CameraProcessor(RGB_URL, DEPTH_URL)
    controller = GestureController()
    camera.start()

    print("Hand Mouse active. Move hand=cursor, Peace=click, Fist=right-click")
    print("Thumb Up=scroll up, Open Palm=scroll down. Press 'q' to quit.")

    try:
        while True:
            rgb = camera.fetch_rgb()
            depth = camera.fetch_depth()
            if rgb is None:
                time.sleep(0.1)
                continue

            depth = CameraProcessor._resize_depth(depth, rgb.shape)
            rgb_mp = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            results = camera.hands.process(rgb_mp)

            output = rgb.copy()
            gesture = "No hand"
            distance = None
            cx = cy = 0

            if results.multi_hand_landmarks:
                hl = results.multi_hand_landmarks[0]
                lm = hl.landmark
                cx = int(lm[8].x * output.shape[1])
                cy = int(lm[8].y * output.shape[0])

                hlabel = "Unknown"
                if results.multi_handedness:
                    hlabel = results.multi_handedness[0].classification[0].label

                states = controller._finger_states(hl.landmark, hlabel)
                gesture = controller.classify(states)
                distance = controller.get_distance(depth, cx, cy)

                # Move cursor with index finger tip
                controller.move_cursor(cx, cy, output.shape[1], output.shape[0])

                # Gesture actions
                controller.handle(gesture, distance, hlabel, cx, cy)
                controller.scroll(distance)

                cv2.circle(output, (cx, cy), 10, (0, 255, 255), -1)
                cv2.putText(output, f"{gesture} | {distance:.2f}m" if distance else gesture,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Hand Mouse (q=quit)", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
