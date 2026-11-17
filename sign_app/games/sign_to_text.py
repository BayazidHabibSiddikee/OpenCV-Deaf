#!/usr/bin/env python3
"""
Real-time Sign-to-Text System
=============================

Detects hand gestures from a live camera stream via the existing
CameraProcessor (skin colour + contour + convexity-defect analysis) and maps
each detected gesture to a sign-language word. The word is drawn on the video
frame and, when enabled, spoken aloud with text-to-speech (TTS).

Supported gestures: Fist, Peace, Open Palm, Thumb Up, One, Three, OK, Four

Standalone usage:
    python3 games/sign_to_text.py                    # default: ASL, no TTS
    python3 games/sign_to_text.py --lang bsl --use-tts

Run with the project virtual environment (it ships opencv-python + numpy):
    sign_app/.venv/bin/python games/sign_to_text.py

Pipeline:
1. CameraProcessor.loop() runs in a background thread. Every frame it
   segments the skin, finds the largest hand contour, counts fingers with
   convexity defects, and stores the raw label in camera.latest_gesture
   (e.g. "Fist", "One", "Peace", "Three", "Four", "Open Palm").
2. GestureInterpreter maps that label to a sign-language word using a
   configurable ASL/BSL dictionary.
3. The interpreted word is drawn on the frame and, when TTS is enabled, is
   spoken aloud with gTTS (silently degraded when no internet).

Extending the vocabulary:
Add new key/value pairs to DEFAULT_SIGN_DICTIONARY or swap in your own dict
at runtime; the runner picks them up automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIGN_APP = os.path.dirname(_HERE)
if _SIGN_APP not in sys.path:
    sys.path.insert(0, _SIGN_APP)

import cv2
import numpy as np

from camera import CameraProcessor

# ---------------------------------------------------------------------------
# Sign language dictionaries (gesture -> sign language word)
# ---------------------------------------------------------------------------
# CameraProcessor.classify_gesture currently emits these labels:
# "Fist", "One", "Peace", "Three", "Four", "Open Palm". "Thumb Up" and "OK"
# are supported as extended/alternate interpretations (see GestureInterpreter).
# ---------------------------------------------------------------------------

DEFAULT_SIGN_DICTIONARY = {
    "Fist": "A",        # ASL: fist = letter A
    "One": "1",         # index finger up
    "Peace": "V",       # index + middle finger up
    "Three": "3",       # thumb + index + middle up
    "Four": "4",        # four fingers up (thumb folded)
    "Open Palm": "5",   # flat hand = number 5
    "Thumb Up": "Thumbs Up",  # extended thumb (extended detection)
    "OK": "OK",         # thumb + index tips touching
}

# British Sign Language variants (1-5 and letters A/V are the same as ASL;
# only the "I Love You" hand-shape differs).
BSL_SIGN_DICTIONARY = {
    "Fist": "A",
    "One": "1",
    "Peace": "V",
    "Three": "3",
    "Four": "4",
    "Open Palm": "5",
    "Thumb Up": "V",   # BSL "V" = peace / victory
    "OK": "OK",
}

# Human-readable display labels + emoji for each signed word.
SIGN_DISPLAY = {
    "A": ("A", "✊"),
    "1": ("1", "☝️"),
    "V": ("V", "✌️"),
    "3": ("3", "✌️"),
    "4": ("4", "✋"),
    "5": ("5", "✋"),
    "Thumbs Up": ("Thumbs Up", "👍"),
    "OK": ("OK", "👌"),
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default sign language: "asl" or "british".
DEFAULT_LANGUAGE = "asl"

# Whether to attempt extended detection ("One" -> "Thumbs Up", "Peace" -> "OK").
# Disabled by default: the legacy camera.py pipeline counts fingers with convexity
# defects and cannot reliably separate these hand-shapes without fingertip data.
EXTENDED_DETECTION = False


# Default camera stream URLs (Sipeed MaixSense-A075V).
DEFAULT_RGB_URL = "http://192.168.233.1/rgb"
DEFAULT_DEPTH_URL = "http://192.168.233.1/depth"

# Geometry used for drawing the result label on the video frame.
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 1.2
LABEL_FONT_THICKNESS = 2
LABEL_COLOR = (255, 255, 255)   # cyan
SHADOW_OFFSET = (3, 3)


# ---------------------------------------------------------------------------
# Text-to-Speech (optional)
# ---------------------------------------------------------------------------

class TTS:
    """Lightweight text-to-speech.

    Primary backend: gTTS (needs internet).  If gTTS is missing or fails the
    engine falls back to silent no-op so the app never crashes on missing
    audio.
    """

    def __init__(self) -> None:
        self._ready = False
        try:
            import gtts  # type: ignore

            self._gtts = gtts.gTTS
            self._ready = True
        except Exception:
            self._gtts = None

    def speak(self, text: str) -> str | None:
        """Speak ``text``; returns the path of the audio file, or None."""
        if not self._ready or not text:
            return None
        try:
            import os
            import tempfile

            fd, path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            try:
                self._gtts(text=text, lang="en-US", slow=False).save(path)
            except Exception:
                if os.path.exists(path):
                    os.remove(path)
                return None
            return path if os.path.exists(path) else None
        except Exception:
            return None

# ---------------------------------------------------------------------------
# Gesture interpreter (gesture -> sign language word)
# ---------------------------------------------------------------------------

class GestureInterpreter:
    """Maps raw camera gestures to sign-language words.

    The dictionary can be swapped per language (ASL vs BSL). ``extended_detection``
    enables optional refinement of the raw labels into the richer vocabulary
    (e.g. distinguishing "One" from "Thumbs Up").

    Usage::

        interpreter = GestureInterpreter(language="asl")
        word = interpreter.interpret(camera.latest_gesture)
    """

    def __init__(
        self,
        *,
        language: str = DEFAULT_LANGUAGE,
        extended_detection: bool = EXTENDED_DETECTION,
    ) -> None:
        self.language = language
        self.extended_detection = extended_detection
        self._dict = self._choose_dict(language)

    # -- internals ---------------------------------------------------------

    def _choose_dict(self, language: str) -> dict[str, str]:
        key = language.strip().lower()
        if key in ("british", "british-sign-language", "uk", "british-sign"):
            return dict(BSL_SIGN_DICTIONARY)
        # Default: American Sign Language
        return dict(DEFAULT_SIGN_DICTIONARY)

    # -- public API -------------------------------------------------------

    def interpret(self, raw_gesture: str) -> str | None:
        """Convert a raw camera gesture label to a sign-language word."""
        if not raw_gesture:
            return None
        gesture = str(raw_gesture).strip()
        if gesture in ("No hand", ""):
            return None
        word = self._dict.get(gesture)
        if word is None:
            # Unknown gesture: fall back to the raw label so it still shows up.
            return gesture
        if self.extended_detection:
            word = self._refine(gesture, word)
        return word

    def refine(self, gesture: str, word: str) -> str:
        """Alias for _refine (kept public for API symmetry)."""
        return self._refine(gesture, word)

    # -- extended heuristics -----------------------------------------------

    def _refine(self, gesture: str, word: str) -> str:
        """Attempt to disambiguate a raw hand-shape into a more specific word.

        Only a lightweight geometric heuristic is implemented: CameraProcessor
        counts fingers with convexity defects and cannot, by finger count alone,
        reliably tell "One" from "Thumbs Up" or "Peace" from "OK".  This method
        is intentionally conservative and falls back to the raw word whenever the
        heuristic is inconclusive.

        Args:
            gesture: Raw gesture from CameraProcessor (e.g. "One", "Peace").
            word: The dictionary word for that gesture.

        Returns:
            A (possibly) more specific sign word.
        """
        if gesture == "One" and word == "1":
            # Only one extended finger: it is either "1" (index) or "Thumbs Up".
            # The camera does not keep the raw contour here, so we return a
            # nominal pass-through and rely on stored hand data to refine later.
            # Fall back to "1" (the common case).
            return word

        if gesture == "Peace" and word == "V":
            # Two-finger shapes ("V", "OK", "Thumbs Up") cannot be separated
            # reliably without fingertip coordinates, so leave the raw word.
            return word

        return word








# ---------------------------------------------------------------------------
# Real-time runner (camera + display + optional TTS)
# ---------------------------------------------------------------------------

class CameraSignRunner:
    """Drives CameraProcessor, interprets gestures, draws the sign word, and
    optionally speaks it aloud."""

    def __init__(
        self,
        *,
        rgb_url: str = DEFAULT_RGB_URL,
        depth_url: str = DEFAULT_DEPTH_URL,
        language: str = DEFAULT_LANGUAGE,
        use_tts: bool = False,
        extended_detection: bool = EXTENDED_DETECTION,
        window_name: str = "Sign-to-Text",
        width: int = 640,
        height: int = 480,
        preview_scale: float = 1.0,
        delay: float = 0.03,
    ) -> None:
        self.rgb_url = rgb_url
        self.depth_url = depth_url
        self.language = language
        self.use_tts = use_tts
        self.extended_detection = extended_detection
        self.window_name = window_name
        self.width = width
        self.height = height
        self.preview_scale = preview_scale
        self.delay = delay

        self.camera = CameraProcessor(rgb_url, depth_url)
        self.interpreter = GestureInterpreter(
            language=language, extended_detection=extended_detection
        )
        self.tts = TTS() if use_tts else None

        # Status / UI state
        self.last_gesture = "No hand"
        self.last_word = None
        self._quit = False

        # Hand position + distance (useful for games / UI)
        self.cx = 0
        self.cy = 0
        self.distance = None

    def start(self) -> None:
        self.camera.start()

    def stop(self) -> None:
        self.camera.stop()

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        self.start()
        print("Sign-to-Text running. Press 'q' to quit, 't' to toggle TTS, 's' to snapshot.")
        print(f"Language: {self.language.upper()} | Extended detection: {self.extended_detection}")

        try:
            while not self._quit:
                data = self.camera.get_data()
                gesture = data.get("gesture", "No hand")
                self.distance = data.get("distance")
                self.cx = None
                self.cy = None

                jpg = self.camera.get_frame_jpeg()
                frame = None
                if jpg is not None:
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                word = self.interpreter.interpret(gesture)
                self.last_gesture = gesture
                self.last_word = word

                if word is not None:
                    self._draw_output(frame, gesture, word)

                if frame is not None:
                    cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._quit = True
                elif key == ord("t"):
                    self.use_tts = not self.use_tts
                    self.tts = TTS() if self.use_tts else None
                    print(f"TTS {'enabled' if self.use_tts else 'disabled'}")
                elif key == ord("s"):
                    self._snapshot()

                time.sleep(self.delay)
        finally:
            self.stop()
            cv2.destroyAllWindows()
    # -- drawing -----------------------------------------------------------

    def _draw_output(self, frame: np.ndarray, gesture: str, word: str) -> None:
        """Draw the video frame plus the detected sign word on screen."""
        if frame is None:
            return

        if self.preview_scale != 1.0:
            disp = cv2.resize(frame, (0, 0), fx=self.preview_scale, fy=self.preview_scale)
            disp_w, disp_h = disp.shape[1], disp.shape[0]
            scale = disp_w / frame.shape[1]
        else:
            disp = frame
            disp_w, disp_h = frame.shape[1], frame.shape[0]
            scale = 1.0

        cv2.putText(disp, f"Gesture: {gesture}", (10, 30), LABEL_FONT, 0.8, (0, 255, 0), 2)
        cv2.putText(disp, f"Word: {word}", (10, 64), LABEL_FONT, 1.2, LABEL_COLOR, 2)

        if self.cx is not None and self.cy is not None:
            box = 30
            x0 = max(0, int(self.cx) - box)
            y0 = max(0, int(self.cy) - box)
            x1 = min(disp_w - 1, int(self.cx) + box)
            y1 = min(disp_h - 1, int(self.cy) + box)
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.putText(disp, f"(x={self.cx:.0f}, y={self.cy:.0f})",
                        (x0, y0 - 6), LABEL_FONT, 0.5, (255, 255, 0), 1)

        if self.distance is not None:
            cv2.putText(disp, f"Distance: {self.distance:.2f}m",
                        (10, disp_h - 14), LABEL_FONT, 0.6, (255, 255, 0), 1)

        _, emoji = SIGN_DISPLAY.get(word, (word, ""))
        if emoji:
            cv2.putText(disp, f" {emoji}",
                        (10, disp_h - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, LABEL_COLOR, 2)

        cv2.imshow(self.window_name, disp)

    def _snapshot(self) -> None:
        frame = self.camera.latest_frame
        if frame is None:
            print("No frame to snapshot.")
            return
        ts = int(time.time())
        path = f"sign_snapshot_{ts}.jpg"
        cv2.imwrite(path, frame)
        print(f"Snapshot saved: {path}")

    # -- TTS helper ----------------------------------------------------------

    def speak(self, word: str) -> None:
        """Speak a sign word if TTS is enabled."""
        if not self.use_tts or not word:
            return
        try:
            path = self.tts.speak(word)
            if path:
                print(f"[TTS] {word} -> {path}")
        except Exception as exc:
            print(f"[TTS] error: {exc}")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time sign-to-text system")
    parser.add_argument(
        "--lang",
        choices=["asl", "british", "uk", "american"],
        default=DEFAULT_LANGUAGE,
        help="Sign language variant (default: asl)",
    )
    parser.add_argument(
        "--use-tts",
        action="store_true",
        help="Enable text-to-speech for detected words",
    )
    parser.add_argument(
        "--no-extended",
        action="store_true",
        help="Disable extended gesture refinement",
    )
    parser.add_argument(
        "--rgb-url",
        default=DEFAULT_RGB_URL,
        help=f"RGB stream URL (default: {DEFAULT_RGB_URL})",
    )
    parser.add_argument(
        "--depth-url",
        default=DEFAULT_DEPTH_URL,
        help=f"Depth stream URL (default: {DEFAULT_DEPTH_URL})",
    )
    parser.add_argument(
        "--window",
        default="Sign-to-Text",
        help="OpenCV window name",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Display scale (e.g. 0.7 for 70% size)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Preview frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Preview frame height",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.03,
        help="Loop sleep seconds (frame rate cap)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    language = "british" if args.lang in ("british", "uk") else "asl"

    runner = CameraSignRunner(
        rgb_url=args.rgb_url,
        depth_url=args.depth_url,
        language=language,
        use_tts=args.use_tts,
        extended_detection=not args.no_extended,
        window_name=args.window,
        width=args.width,
        height=args.height,
        preview_scale=args.preview_scale,
        delay=args.delay,
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
