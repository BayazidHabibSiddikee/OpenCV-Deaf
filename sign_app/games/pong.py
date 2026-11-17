#!/usr/bin/env python3
"""
Hand-Controlled Pong (tkinter, no extra deps)
Move the paddle with your hand's Y position. Peace gesture launches the ball.

Run: python3 games/pong.py
"""

import sys
import time
import tkinter as tk
import cv2
import numpy as np

sys.path.insert(0, "/home/bayazid/Documents/OpenCV/sign_app")
from camera import CameraProcessor
from gesture_controller import GestureController

RGB_URL = "http://192.168.233.1/rgb"
DEPTH_URL = "http://192.168.233.1/depth"

W, H = 700, 500
PADDLE_W, PADDLE_H = 15, 90
BALL_SIZE = 14


class PongGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hand Pong")
        self.canvas = tk.Canvas(self.root, width=W, height=H, bg="black")
        self.canvas.pack()

        self.paddle_y = H // 2 - PADDLE_H // 2
        self.ball_x = W // 2
        self.ball_y = H // 2
        self.ball_dx = 5
        self.ball_dy = 4
        self.score = 0
        self.running = False

        self.paddle = self.canvas.create_rectangle(
            20, self.paddle_y, 20 + PADDLE_W, self.paddle_y + PADDLE_H, fill="#22d3ee"
        )
        self.ball = self.canvas.create_oval(
            self.ball_x, self.ball_y,
            self.ball_x + BALL_SIZE, self.ball_y + BALL_SIZE,
            fill="#ef4444",
        )
        self.score_text = self.canvas.create_text(
            W - 60, 30, text=f"Score: {self.score}", fill="white", font=("Arial", 20)
        )
        self.start_text = self.canvas.create_text(
            W // 2, H // 2, text="Peace gesture to start", fill="white", font=("Arial", 24)
        )

    def update_paddle(self, cy, frame_h):
        # Map hand Y to paddle Y, clamped
        py = int((cy / frame_h) * H) - PADDLE_H // 2
        py = max(0, min(H - PADDLE_H, py))
        self.paddle_y = py

    def launch(self):
        if not self.running:
            self.running = True
            self.canvas.itemconfig(self.start_text, text="")

    def step(self):
        if not self.running:
            self.root.after(16, self.step)
            return

        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Top/bottom bounce
        if self.ball_y <= 0 or self.ball_y + BALL_SIZE >= H:
            self.ball_dy *= -1

        # Left wall (miss)
        if self.ball_x <= 0:
            self.ball_dx *= -1
            self.score = max(0, self.score - 1)

        # Paddle hit
        if (
            self.ball_x <= 20 + PADDLE_W
            and self.paddle_y <= self.ball_y + BALL_SIZE
            and self.ball_y <= self.paddle_y + PADDLE_H
        ):
            self.ball_dx *= -1
            self.score += 1
            self.canvas.itemconfig(self.score_text, text=f"Score: {self.score}")

        # Right wall (passed paddle)
        if self.ball_x >= W:
            self.ball_dx *= -1

        self.canvas.coords(
            self.paddle, 20, self.paddle_y, 20 + PADDLE_W, self.paddle_y + PADDLE_H
        )
        self.canvas.coords(
            self.ball,
            self.ball_x, self.ball_y, self.ball_x + BALL_SIZE, self.ball_y + BALL_SIZE,
        )

        self.root.after(16, self.step)


def main():
    camera = CameraProcessor(RGB_URL, DEPTH_URL)
    controller = GestureController()
    camera.start()

    game = PongGame()
    last_gesture = None

    print("Hand Pong active. Move hand Y=paddle, Peace=launch ball. Close window to quit.")

    def loop():
        nonlocal last_gesture
        try:
            rgb = camera.fetch_rgb()
            depth = camera.fetch_depth()
            if rgb is not None:
                depth = CameraProcessor._resize_depth(depth, rgb.shape)
                rgb_mp = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                results = camera.hands.process(rgb_mp)

                if results.multi_hand_landmarks:
                    hl = results.multi_hand_landmarks[0]
                    lm = hl.landmark
                    cy = int(lm[8].y * rgb.shape[0])

                    hlabel = "Unknown"
                    if results.multi_handedness:
                        hlabel = results.multi_handedness[0].classification[0].label

                    states = controller._finger_states(hl.landmark, hlabel)
                    gesture = controller.classify(states)

                    game.update_paddle(cy, rgb.shape[0])

                    if gesture == "Peace" and gesture != last_gesture:
                        game.launch()
                    last_gesture = gesture

            game.step()
        except tk.TclError:
            # Window closed
            camera.stop()
            return

        # Poll at ~30fps
        game.root.after(33, loop)

    game.root.after(33, loop)
    game.root.mainloop()
    camera.stop()


if __name__ == "__main__":
    main()
