"""Simple Sign Language Detection - Emotion + Hand Signs."""
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse

from camera import CameraProcessor

app = FastAPI(title="Sign Language Detection")

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
template_dir = BASE_DIR / "templates"

LESSONS = [
    {"name": "Hello", "emoji": "👋", "gesture": "Open Palm", "hint": "Show an open palm", "dist": [0.5, 1.5]},
    {"name": "Yes", "emoji": "👍", "gesture": "Thumb Up", "hint": "Thumbs up", "dist": [0.5, 1.5]},
    {"name": "No", "emoji": "✊", "gesture": "Fist", "hint": "Make a fist", "dist": [0.5, 1.5]},
    {"name": "Peace", "emoji": "✌️", "gesture": "Peace", "hint": "Index and middle finger", "dist": [0.5, 1.5]},
    {"name": "One", "emoji": "☝️", "gesture": "One", "hint": "Index finger only", "dist": [0.5, 1.5]},
    {"name": "OK", "emoji": "👌", "gesture": "OK", "hint": "Thumb and index touch", "dist": [0.4, 1.2]},
]

@app.on_event("startup")
def _startup():
    pass

@app.on_event("shutdown")
def _shutdown():
    pass

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory=template_dir)
    return templates.TemplateResponse("index_minimal.html", {"request": request, "lessons": LESSONS})

@app.get("/video_feed")
async def video_feed():
    def generate():
        while True:
            jpg = CameraProcessor(
                rgb_url="http://192.168.233.1/rgb",
                depth_url="http://192.168.233.1/depth",
            ).get_frame_jpeg()
            if jpg is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n\n" + jpg + b"\r\n")
            time.sleep(0.03)

    return StreamingResponse(generate(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/gesture_data")
async def gesture_data():
    return CameraProcessor(
        rgb_url="http://192.168.233.1/rgb",
        depth_url="http://192.168.233.1/depth",
    ).get_data()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
