"""
Sign Language Detection & Communication Platform
Integrates:
- Real-time ASL fingerspelling recognition (SignFrame AI PyTorch CNN)
- Prediction stabilization & text composition
- Assistive Text-to-Speech (TTS)
- 3D ToF foreground isolation & facial emotion detection
"""

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Setup module paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from camera import CameraProcessor

app = FastAPI(title="Sign Language Communication & Teacher Platform")
template_dir = BASE_DIR / "templates"
templates = Jinja2Templates(directory=template_dir)

# Global camera processor singleton
processor = CameraProcessor()

LESSONS = [
    {"name": "A", "emoji": "🅰️", "gesture": "A", "hint": "Fist with thumb resting alongside index finger", "dist": [0.4, 1.5]},
    {"name": "B", "emoji": "🅱️", "gesture": "B", "hint": "Open flat palm facing forward with thumb tucked across palm", "dist": [0.4, 1.5]},
    {"name": "C", "emoji": "🅲", "gesture": "C", "hint": "Curved hand forming a C shape", "dist": [0.4, 1.5]},
    {"name": "D", "emoji": "🅳", "gesture": "D", "hint": "Index pointing up, other fingers and thumb touch forming a circle", "dist": [0.4, 1.5]},
    {"name": "L", "emoji": "🇱", "gesture": "L", "hint": "Thumb and index extended at right angle forming an L", "dist": [0.4, 1.5]},
    {"name": "V", "emoji": "✌️", "gesture": "V", "hint": "Index and middle extended in a V victory sign", "dist": [0.4, 1.5]},
    {"name": "W", "emoji": "🇼", "gesture": "W", "hint": "Index, middle, and ring fingers extended upward", "dist": [0.4, 1.5]},
    {"name": "Y", "emoji": "🤙", "gesture": "Y", "hint": "Thumb and pinky extended out, middle three curled down", "dist": [0.4, 1.5]},
]


@app.on_event("startup")
def _startup():
    processor.start()


@app.on_event("shutdown")
def _shutdown():
    processor.stop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "lessons": LESSONS,
    })


@app.get("/video_feed")
async def video_feed():
    def generate():
        while True:
            jpg = processor.get_frame_jpeg()
            if jpg is None:
                time.sleep(0.04)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/gesture_data")
async def gesture_data():
    return JSONResponse(processor.get_data())


@app.post("/compose/add_letter")
async def compose_add_letter():
    text = processor.add_stable_letter()
    return JSONResponse({"text": text, "status": "added"})


@app.post("/compose/space")
async def compose_space():
    text = processor.add_space()
    return JSONResponse({"text": text, "status": "space"})


@app.post("/compose/backspace")
async def compose_backspace():
    text = processor.backspace()
    return JSONResponse({"text": text, "status": "backspace"})


@app.post("/compose/clear")
async def compose_clear():
    text = processor.clear_text()
    return JSONResponse({"text": text, "status": "cleared"})


@app.post("/compose/speak")
async def compose_speak():
    ok, msg = processor.speak_text()
    return JSONResponse({"ok": ok, "message": msg, "text": processor.composed_text})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
