# OpenCV-Deaf & SignFrame AI Vision Suite

Unified Computer Vision & Deep Learning suite for American Sign Language (ASL) fingerspelling, 3D hand tracking, metric depth estimation, and expressive gesture communication. Designed for standard webcams and the **Sipeed MaixSense-A075V** RGB-D Time-of-Flight (ToF) camera module.

---

## 🌟 Core Features & Modules

### 1. SignFrame AI (ASL Fingerspelling & Speech)
- **24 Static ASL Letters:** Real-time fingerspelling recognition powered by a custom PyTorch CNN (`models/feedback_candidate.pth`).
- **Prediction Stabilization:** Temporal smoothing across frames before letter confirmation to prevent flickering.
- **Text Composition & TTS:** Build words and sentences live, edit messages, and read aloud using Text-to-Speech.
- **Run Webcam App:**
  ```bash
  python3 src/webcam_app.py
  ```

### 2. MaixSense-A075V 3D ToF Vision Studios
- **3D Depth Foreground Isolation:** Locks onto hands in 3D millimeters (10 cm – 1.8 m), completely eliminating background bleed (walls, furniture, curtains).
- **Hugging Face SigLIP-2 Zero-Shot Recognition:** Flagship vision-language model (`google/siglip2-so400m-patch14-384`) classifying 14+ expressive gestures and 8 real-life facial emotions.
- **Multi-Hand 3D Engine:** Tracks 1 to 4 hands simultaneously using MediaPipe 21 3D landmarks with real-time metric distance telemetry.
- **FastAPI Web Teacher:** Interactive sign language learning web app with live camera feed and scoring.

---

## 📁 Repository Structure

```text
OpenCV/
├── src/                        # SignFrame AI core modules
│   ├── webcam_app.py           # Main interactive ASL fingerspelling & TTS app
│   ├── model.py                # PyTorch CNN architecture for ASL
│   ├── inference.py            # Real-time inference engine
│   ├── hand_tracker.py         # MediaPipe hand tracking & cropping
│   ├── stabilizer.py           # Temporal prediction smoothing
│   ├── text_composer.py        # Sentence builder & message editing
│   ├── communication.py        # Text-to-Speech (TTS) integration
│   ├── preprocessing.py        # Grayscale & 28x28 normalization
│   ├── dataset.py              # Dataset loading & PyTorch DataLoaders
│   ├── train.py                # Training & fine-tuning pipeline
│   ├── evaluate.py             # Evaluation & confusion matrix generator
│   └── visualizer.py           # Frame annotations & UI overlay
│
├── models/
│   └── feedback_candidate.pth  # Pretrained ASL CNN checkpoint (24 classes)
│
├── hf_siglip_studio.py         # Hugging Face SigLIP-2 (3.5GB) Zero-Shot Gesture & Emotion Studio
├── multi_hand_studio.py        # Multi-Hand (1-4 hands) 3D Studio + ToF Depth Telemetry
├── a075v_vision_studio.py      # All-in-One Vision Studio with ToF Depth Isolation
├── hand_gesture_depth.py       # Standalone 3D Depth Hand Gesture & Distance Monitor
├── hand_gesture_mediapipe.py   # MediaPipe 3D Landmark Demo
│
├── sign_app/                   # FastAPI Sign Language Teacher Web Application
│   ├── camera.py               # A075V composite binary packet decoder & depth slicer
│   ├── emotion_detector.py     # YuNet + FER+ emotion classifier
│   └── main.py                 # FastAPI backend endpoints
│
├── docs/
│   └── PROJECT_SPEC.md         # SignFrame AI technical specification
├── results/                    # Confusion matrices and evaluation metrics
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start

### 1. Environment Setup
Install the dependencies inside your Python environment:
```bash
pip install -r requirements.txt
pip install opencv-contrib-python mediapipe requests onnxruntime pyautogui
```

### 2. Running Applications

#### A. ASL Fingerspelling & Text-to-Speech (Webcam)
```bash
python3 src/webcam_app.py
```
- Hold an ASL hand sign in front of your camera.
- The stabilized prediction confirms letters into the message box.
- Press **Space** for spaces, **Backspace** to delete, and **Enter/Speak** to vocalize.

#### B. Hugging Face SigLIP-2 Vision Studio (A075V / ToF Camera)
```bash
python3 hf_siglip_studio.py
```
- Recognizes 14+ expressive gestures and 8 facial emotions zero-shot.
- Metric millimeter depth estimation on your hands and body.

#### C. Multi-Hand 3D Studio (MediaPipe + Depth)
```bash
python3 multi_hand_studio.py
```

#### D. FastAPI Sign Language Learning Platform
```bash
cd sign_app
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.
