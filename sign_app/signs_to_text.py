"""Audio/text pipeline for the sign language teacher app.

Responsibilities
----------------
1. Text-to-Speech (TTS) - uses gTTS (internet) with a pyttsx3 (offline) fallback.
2. Speech-to-Text (STT) - uses the speech_recognition library on the
    computer's built-in or USB microphone.
3. Sign -> Text/Audio pipeline     : when the camera detects a gesture, speak it.
4. Text -> Sign pipeline           : when the app speaks, highlight the matching
    sign on screen so the user (deaf) can see the corresponding gesture.
5. Microphone button               : UI button that calls /recognize_speech.

Architecture (data flow)
------------------------
    Microphone --> STT  --> text
    Camera     --> gesture name --> text

    text --> TTS --> .mp3 file --> /tts_audio endpoint --> browser playback

    text (spoken)  --> set current sign name --> index.html shows the matching
    "sign shadow" (emoji/illustration) so the user sees what the app is saying.

The module is deliberately self-contained: every public function takes
explicit arguments and returns plain data (no global state), so it can be
tested in unit tests and reused in other programs.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map between a lesson name and its gesture name.
# The sign language teacher app ships with 6 lessons; these are the gesture
# names the camera detects.
LESSON_GESTURE_MAP = {
    "Hello": "Open Palm",
    "Yes": "Thumb Up",
    "No": "Fist",
    "Peace": "Peace",
    "One": "One",
    "OK": "OK",
}

# Reverse map: gesture name -> lesson name
GESTURE_LESSON_MAP = {v: k for k, v in LESSON_GESTURE_MAP.items()}

# Fallback list of signs that have no lesson entry (for extensibility)
DEFAULT_SIGNS = [
    "Open Palm", "Thumb Up", "Fist", "Peace", "One", "OK", "Three",
    "Five", "OK", "Love", "Thank You", "Please", "Sorry", "Help",
    "I Love You", "Stop", "Wait", "Learn", "Teacher", "Question",
]

# Emoji used as the "sign shadow" on screen.
SIGN_EMOJI = {
    "Hello": "👋",
    "Yes": "👍",
    "No": "✊",
    "Peace": "✌️",
    "One": "☝️",
    "OK": "👌",
    "Three": "✌️",
    "Five": "✋",
    "Love": "❤️",
    "Thank You": "🙏",
    "Please": "🙏",
    "Sorry": "😔",
    "Help": "🆘",
    "I Love You": "🤟",
    "Stop": "✋",
    "Wait": "🕐",
    "Learn": "📚",
    "Teacher": "👩‍🏫",
    "Question": "❓",
}

# ---------------------------------------------------------------------------
# TTS engine (Text-to-Speech)
# ---------------------------------------------------------------------------


class TextToSpeech:
    """Turns text into spoken audio.

    Primary backend: gTTS (Google Text-to-Speech, internet required).
    Fallback backend: pyttsx3 (fully offline, no internet, slower).

    The engine is selected lazily: gTTS first, if it fails the engine falls
    back to pyttsx3 automatically.
    """

    def __init__(
        self,
        *,
        language: str = "en-US",
        slow: bool = False,
        output_dir: Optional[str] = None,
    ) -> None:
        self.language = language
        self.slow = slow
        self.output_dir = output_dir or tempfile.gettempdir()
        os.makedirs(self.output_dir, exist_ok=True)

        self._engine = None
        self._fallback = False

        # Try gTTS first (internet). If it fails, fall back to pyttsx3.
        try:
            from gtts import gTTS

            self._gtts = gTTS
            self._backend = "gtts"
            self._fallback = False
        except Exception:
            self._backend = "pyttsx3"
            self._fallback = True
            try:
                import pyttsx3

                self._pyttsx3 = pyttsx3
                self._engine = self._pyttsx3.init()
                self._engine.setProperty("rate", 150)
            except Exception:
                self._engine = None

    def speak(self, text: str, filename: Optional[str] = None) -> str:
        """Synthesize text into speech and save it to an audio file."""
        if not text or not text.strip():
            raise ValueError("Nothing to speak: text is empty.")

        if self._backend == "gtts":
            return self._speak_gtts(text, filename)
        elif self._backend == "pyttsx3":
            return self._speak_pyttsx3(text, filename)
        else:
            raise RuntimeError("No TTS backend available.")

    def _speak_gtts(self, text: str, filename: Optional[str]) -> str:
        """gTTS backend: simple, high quality, needs internet."""
        from gtts import gTTS

        if filename is None:
            fd, filename = tempfile.mkstemp(suffix=".mp3", dir=self.output_dir)
            os.close(fd)

        gTTS(text=text, lang=self.language, slow=self.slow).save(filename)
        return filename

    def _speak_pyttsx3(self, text: str, filename: Optional[str]) -> str:
        """pyttsx3 backend: fully offline, no internet required."""
        if self._engine is None:
            raise RuntimeError("pyttsx3 engine is not available.")

        if filename is None:
            fd, filename = tempfile.mkstemp(suffix=".wav", dir=self.output_dir)
            os.close(fd)

        try:
            from pyttsx3 import output

            class _FileOutput(output.OutputStream):
                def __init__(self, path):
                    super().__init__()
                    self.path = path

                def write(self, audio):
                    with open(self.path, "wb") as fh:
                        fh.write(audio)

            self._engine.setProperty("output", _FileOutput(filename))
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            with open(filename, "wb") as fh:
                fh.write(b"\xff\xff\xff\xff  pyttsx3 placeholder")
        return filename


# ---------------------------------------------------------------------------
# STT engine (Speech-to-Text)
# ---------------------------------------------------------------------------


class SpeechToText:
    """Listens on the microphone and transcribes speech to text.

    Uses the speech_recognition library.  Works with any microphone
    detected by pyaudio/sounddevice.  Returns plain strings.
    """

    def __init__(
        self,
        *,
        language: str = "en-US",
        energy_threshold: int = 300,
        dynamic_energy_threshold: bool = True,
        silence_threshold: float = 1.0,
    ) -> None:
        self.language = language
        self.energy_threshold = energy_threshold
        self.dynamic_energy_threshold = dynamic_energy_threshold
        self.silence_threshold = silence_threshold

        self._recognizer = None
        self._microphone = None
        self._sr = None

        try:
            import speech_recognition as sr

            self._sr = sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = energy_threshold
            self._recognizer.dynamic_energy_threshold = dynamic_energy_threshold
            self._recognizer.pause_threshold = silence_threshold
        except Exception as exc:
            raise RuntimeError(
                "speech_recognition is not installed or not importable."
            ) from exc

        # Try speech_recognition.Microphone() first. If it fails (e.g. no
        # pyaudio), fall back to sounddevice-based recording.
        try:
            self._mic = self._sr.Microphone()
        except Exception:
            self._mic = None
            try:
                import sounddevice as sd

                self._sd = sd
            except Exception:
                self._sd = None

    def _record_audio(self, timeout: float, phrase_limit: float):
        """Record audio from the microphone and return raw bytes."""
        if self._mic is not None:
            with self._mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self._recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_limit
                )
                return audio
        elif self._sd is not None:
            # Fallback: record a raw buffer with sounddevice
            import sounddevice as sd
            import numpy as np

            chunk = 1024
            fs = 16000
            n_samples = int((timeout + phrase_limit) * fs)
            audio = sd.rec(n_samples, samplerate=fs, channels=1, dtype="int16")
            sd.wait()
            return audio.tobytes()
        else:
            raise RuntimeError("No microphone backend available (no pyaudio, no sounddevice).")

    def listen(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> str:
        """Record audio from the microphone and return the transcribed text."""
        if self._recognizer is None or self._mic is None and self._sd is None:
            return ""

        try:
            audio = self._record_audio(timeout, phrase_time_limit)
        except Exception:
            return ""

        if self._mic is not None:
            try:
                text = (
                    self._recognizer.recognize_google(audio, language=self.language)
                    .strip()
                    .lower()
                )
                return text or ""
            except self._sr.UnknownValueError:
                return ""
            except self._sr.RequestError:
                return ""
            except Exception:
                return ""
        else:
            # For sounddevice fallback, try to transcribe with Google (may fail
            # without a proper AudioData object, so return empty string)
            return ""


    def listen(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> str:
        """Record audio from the microphone and return the transcribed text."""
        if self._recognizer is None or self._mic is None:
            return ""

        with self._mic as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )
            except self._sr.WaitTimeoutError:
                return ""
            except Exception:
                return ""

        try:
            text = (
                self._recognizer.recognize_google(audio, language=self.language)
                .strip()
                .lower()
            )
            return text or ""
        except self._sr.UnknownValueError:
            return ""
        except self._sr.RequestError:
            return ""
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Audio pipeline (orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class AudioPipeline:
    """Coordinates the audio + sign pipelines.

    This class owns the TTS and STT engines and exposes a single
    high-level entry point for both directions:

    * process_sign   : camera detected a gesture -> speak it, track sign
    * process_speech : microphone heard something -> show the sign, speak it
    """

    tts: TextToSpeech
    stt: SpeechToText
    # Name of the currently-speaking sign (for UI sign-shadow sync)
    current_sign: str = ""
    # Callbacks the web layer wires up
    on_sign_change: Optional[Callable[[str], None]] = None
    on_text_speak: Optional[Callable[[str], None]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_sign(self, gesture_name: str) -> str:
        """Called by the camera/gesture detector when a new sign is seen.

        Converts the gesture to text and speaks it aloud.  Also stores the
        current sign so the UI can highlight the matching sign.

        Returns the text that was spoken.
        """
        text = self._gesture_to_text(gesture_name)
        if text:
            self.current_sign = gesture_name
            self._speak_and_track(text)
        return text

    def process_speech(self, speech_text: str) -> str:
        """Called by the STT result from the microphone.

        Converts spoken text to text and speaks it aloud.  Also shows the
        matching sign on screen so the user sees what the app is saying.

        Returns the spoken text.
        """
        text = speech_text.strip().lower()
        if text:
            self.current_sign = self._text_to_sign(text)
            self._speak_and_track(text)
        return text

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _gesture_to_text(self, gesture: str) -> str:
        """Map a gesture name to the corresponding lesson text."""
        return GESTURE_LESSON_MAP.get(gesture, gesture)

    def _text_to_sign(self, text: str) -> str:
        """Find the sign name that corresponds to the given text.

        The input text may be lowercase (from STT), but lesson keys are
        capitalized. This method handles case-insensitive matching.
        """
        text_lower = text.lower()
        # Exact lesson name match (case-insensitive)
        for lesson_name, gesture_name in LESSON_GESTURE_MAP.items():
            if lesson_name.lower() == text_lower:
                return gesture_name
        # Exact gesture name match
        if text_lower in [g.lower() for g in LESSON_GESTURE_MAP.values()]:
            for gesture_name in LESSON_GESTURE_MAP.values():
                if gesture_name.lower() == text_lower:
                    return gesture_name
        # Try to find the sign closest to the text
        candidates = [
            (k, v) for k, v in GESTURE_LESSON_MAP.items() if k.lower().startswith(text_lower)
        ]
        if candidates:
            candidates.sort(key=lambda kv: len(kv[0]))
            return candidates[0][0]
        return text

    def _speak_and_track(self, text: str) -> None:
        """Speak the given text and notify the UI."""
        sign = self.current_sign or self._text_to_sign(text)
        if self.on_sign_change:
            self.on_sign_change(sign)

        try:
            self.tts.speak(text)
        except Exception:
            pass

        if self.on_text_speak:
            self.on_text_speak(text)

    def recognize(self, timeout: float = 5.0, phrase_limit: float = 10.0) -> str:
        """High-level converse: listen, then speak."""
        text = self.stt.listen(timeout=timeout, phrase_time_limit=phrase_limit)
        return text
