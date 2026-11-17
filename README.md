# OpenCV-Deaf & MaixSense-A075V AI Vision Suite

Real-time sign language recognition, 3D hand tracking, metric depth distance estimation, and expressive gesture control designed for the **Sipeed MaixSense-A075V** RGB-D Time-of-Flight (ToF) camera module.

## Features

- **3D ToF Depth Foreground Isolation:** Locks onto hands in 3D millimeters (10 cm – 1.8 m), completely eliminating background bleed (walls, furniture, curtains).
- **Expressive 14-Gesture Classifier:** Real-time recognition of `Hi / Hello`, `Pinch [DRAWING]`, `Fist [ERASE]`, `Peace / Victory`, `Rock / Horns`, `I Love You`, `Call Me`, `Yes / Like`, `No / Dislike`, and more.
- **Pinch-to-Draw Canvas:** Smooth air-drawing triggered when the thumb tip and index tip touch, complete with EMA trajectory smoothing.
- **Face & Emotion Detection:** Lightweight YuNet face detector + FER+ emotion recognition with face/body metric distance telemetry.
- **Sign Language Learning Web App:** FastAPI web interface with interactive sign lessons, speech-to-text, and TTS feedback.
- **Game Controller:** Low-latency PyAutoGUI integration for gesture-controlled games (Dino, Pong, Mouse cursor, Gesture Typer).

## Repository Structure

```text
├── a075v_vision_studio.py     # All-in-One AI Vision Studio (Draw + Emotion + Games + Depth)
├── multi_hand_studio.py       # Multi-Hand (1-4 hands) Expressive Gesture & Pinch-Draw Studio
├── hand_gesture_depth.py      # Standalone 3D Depth Hand Gesture & Distance Monitor
├── hand_gesture_mediapipe.py  # MediaPipe Hand Landmark Detector
├── hf_siglip_studio.py        # Hugging Face SigLIP-2 Zero-Shot Gesture & Emotion Studio
└── sign_app/                  # FastAPI Sign Language Teacher Web Application
    ├── camera.py              # A075V packet decoder & processing pipeline
    ├── emotion_detector.py    # YuNet + FER+ ONNX runtime emotion detector
    ├── main.py                # FastAPI server endpoints
    ├── games/                 # Gesture games (dino, mouse, pong, typing)
    └── templates/             # Web UI templates
```

## Quick Start

### 1. Requirements
Ensure your Python environment has OpenCV, NumPy, MediaPipe, and Requests installed:
```bash
pip install opencv-contrib-python mediapipe numpy requests onnxruntime pyautogui
```

### 2. Camera Setup
Connect the Sipeed MaixSense-A075V via USB. The device creates an RNDIS virtual network interface at `192.168.233.1`. Assign your host machine an IP in the same subnet (e.g., `192.168.233.100/24`).

### 3. Run Applications

**Multi-Hand Expressive Studio:**
```bash
python3 multi_hand_studio.py
```

**Vision Studio (Hand + Air Draw + Emotion + Games):**
```bash
python3 a075v_vision_studio.py
```

**FastAPI Sign Language Teacher:**
```bash
cd sign_app
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
