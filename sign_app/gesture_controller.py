"""Gesture-to-pyautogui controller for local games.

Safety: pyautogui.FAILSAFE is ON — fling your mouse to a screen corner to abort.
Run these scripts on the same Linux box that hosts the camera (X11/Wayland).
"""

import time
import pyautogui

# Disable pyautogui's built-in pause between calls for snappy games
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = True

# Safety cooldown so one gesture doesn't spam keys
COOLDOWN = 0.6


class GestureController:
    def __init__(self, screen_w=None, screen_h=None):
        self.screen_w, self.screen_h = pyautogui.size()
        self.last_action_time = 0
        self.last_gesture = None

        # Map gesture -> action name
        self.actions = {
            "Open Palm": "jump",
            "Fist": "duck",
            "Peace": "click",
            "Thumb Up": "pause",
            "One": "key_1",
            "Three": "key_3",
            "OK": "ok",
        }

    # ------------------------------------------------------------------ #
    def _can_act(self):
        now = time.time()
        if now - self.last_action_time < COOLDOWN:
            return False
        self.last_action_time = now
        return True

    # ------------------------------------------------------------------ #
    def handle(self, gesture, distance, handedness, cx, cy):
        """Called every frame with latest detection. Returns action string."""
        if gesture == "No hand" or gesture == self.last_gesture:
            self.last_gesture = gesture
            return None

        if not self._can_act():
            return None

        self.last_gesture = gesture
        action = self.actions.get(gesture)
        if not action:
            return None

        # Execute the mapped pyautogui action
        self._execute(action, cx, cy)
        return action

    def _execute(self, action, cx, cy):
        if action == "jump":
            pyautogui.press("space")
        elif action == "duck":
            pyautogui.press("down")
        elif action == "click":
            pyautogui.click()
        elif action == "pause":
            pyautogui.press("p")
        elif action == "key_1":
            pyautogui.press("1")
        elif action == "key_3":
            pyautogui.press("3")
        elif action == "ok":
            pyautogui.keyDown("shift")
            pyautogui.keyUp("shift")

    # ------------------------------------------------------------------ #
    def move_cursor(self, cx, cy, frame_w, frame_h):
        """Map hand pixel position to screen position and move mouse."""
        sx = int((cx / frame_w) * self.screen_w)
        sy = int((cy / frame_h) * self.screen_h)
        pyautogui.moveTo(sx, sy)

    def scroll(self, distance):
        """Use hand depth to scroll: near = up, far = down."""
        if distance is None:
            return
        if distance < 0.6:
            pyautogui.scroll(3)
        elif distance > 1.4:
            pyautogui.scroll(-3)
