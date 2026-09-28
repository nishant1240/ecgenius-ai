"""
Voice Assistant module: speech-to-text (faster-whisper) and text-to-speech
(pyttsx3 offline, or edge-tts online). Everything degrades gracefully: if the
optional package isn't installed, the app tells the user what to `pip install`
instead of crashing.
"""
import os
import tempfile


def transcribe_audio(file_bytes: bytes, filename: str = "input.wav"):
    """Returns (text, error). Requires: pip install faster-whisper"""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None, "Speech-to-text needs an extra package. Install it with:\n  pip install faster-whisper"

    tmp_path = os.path.join(tempfile.gettempdir(), filename)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(tmp_path)
        text = " ".join(seg.text for seg in segments).strip()
        return text, None
    except Exception as e:
        return None, f"Transcription failed: {e}"
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def speak_text(text: str, lang: str = "en") -> tuple:
    """Generate speech audio for `text`. Returns (audio_file_path, error).

    Tries pyttsx3 (fully offline) first, falls back to edge-tts (needs
    internet) if pyttsx3 isn't installed. Neither is a hard dependency.
    """
    out_path = os.path.join(tempfile.gettempdir(), "ecgenius_tts.mp3")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        wav_path = os.path.join(tempfile.gettempdir(), "ecgenius_tts.wav")
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        return wav_path, None
    except ImportError:
        pass
    except Exception as e:
        return None, f"pyttsx3 failed: {e}"

    try:
        import asyncio
        import edge_tts
        voice = "hi-IN-SwaraNeural" if lang == "hi" else "en-US-AriaNeural"

        async def _run():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(out_path)

        asyncio.run(_run())
        return out_path, None
    except ImportError:
        return None, ("Text-to-speech needs an extra package. Install one of:\n"
                       "  pip install pyttsx3        (offline)\n"
                       "  pip install edge-tts        (online, higher quality)")
    except Exception as e:
        return None, f"edge-tts failed: {e}"
