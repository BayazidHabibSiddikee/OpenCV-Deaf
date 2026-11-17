#!/usr/bin/env python3
"""
Gesture Typing Game
Hold gestures to type letters into any focused text field.

Gestures -> Letters:
  One       -> A
  Peace     -> B
  Three     -> C
  Thumb Up  -> D
  Open Palm -> E
  Fist      -> Space
  OK        -> Backspace

Run: python3 games/type_gesture.py   (click into a text box first)
"""

import sys
import time
import cv2
import numpy as np

sys.path.insert(0, "/home/bayazid/Documents/OpenCV/sign_app")
from camera import CameraProcessor
from gesture_controller import GestureController

RGB_URL = "http://192.168.233.1/rgb"
DEPTH_URL = "http://192.168.233.1/depth"

GESTURE_KEYS = {
    "One": "a",
    "Peace": "b",
    "Three": "c",
    "Thumb Up": "d",
    "Open Palm": "e",
    "Fist": " ",
    "OK": "\b",
}


def main():
    import pyautogui
    camera = CameraProcessor(RGB_URL, DEPTH_URL)
    controller = GestureController()
    camera.start()

    print("Gesture Typer active. Click into a text field, then hold gestures to type.")
    print("Gestures: One=A, Peace=B, Three=C, ThumbUp=D, OpenPalm=E, Fist=Space, OK=Backspace")
    print("Press 'q' to quit.")

    last_key = None
    key_down_time = 0
    KEY_HOLD = 0.8  # seconds to hold a gesture before typing

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

            if results.multi_hand_landmarks:
                hl = results.multi_hand_landmarks[0]
                lm = hl.landmark
                hlabel = "Unknown"
                if results.multi_handedness:
                    hlabel = results.multi_handedness[0].classification[0].label

                states = controller._finger_states(hl.landmark, hlabel)
                gesture = controller.classify(states)
                cx = int(lm[8].x * output.shape[1])
                cy = int(lm[8].y * output.shape[0])
                cv2.circle(output, (cx, cy), 8, (0, 255, 255), -1)

                key = GESTURE_KEYS.get(gesture)
                now = time.time()

                if key and gesture == last_key:
                    if now - key_down_time > KEY_HOLD:
                        if key == "\b":
                            pyautogui.press("backspace")
                        else:
                            pyautogui.press(key)
                        key_down_time = now  # reset so it repeats every KEY_HOLD
                else:
                    last_key = gesture
                    key_down_time = now

                cv2.putText(output, f"{gesture} -> {key or '-'}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Gesture Typer (q=quit)", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
