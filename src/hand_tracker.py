
from pathlib import Path
import os
import cv2
import mediapipe as mp


def resolve_model_path():
    p1 = Path("assets/hand_landmarker.task")
    if p1.exists():
        return p1
    p2 = Path(os.path.expanduser("~/.opencv/models/hand_landmarker.task"))
    if p2.exists():
        return p2
    return p1


MODEL_PATH = resolve_model_path()


class HandTracker:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing MediaPipe model: {MODEL_PATH}"
            )

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = HandLandmarker.create_from_options(
            options
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        return result.hand_landmarks[0]

    def close(self):
        self.landmarker.close()
