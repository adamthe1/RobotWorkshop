"""
Speech-to-text helper using ElevenLabs.

Requires system PortAudio and Python deps when using voice mode:
  - System: PortAudio (e.g., libportaudio2/portaudio19-dev on Debian/Ubuntu, brew install portaudio on macOS)
  - Python: sounddevice, soundfile, numpy, requests

Env:
  ELEVENLABS_API_KEY=<your key>
"""

import io
import os
import requests


def _lazy_import_audio():
    try:
        import numpy as np  # noqa: F401
        import sounddevice as sd
        import soundfile as sf
        return sd, sf
    except Exception as e:
        raise RuntimeError(
            "Audio stack unavailable (PortAudio likely missing). "
            "Install system PortAudio and Python deps."
        ) from e


def record_wav_bytes(seconds: int = 4, samplerate: int = 16000) -> bytes:
    sd, sf = _lazy_import_audio()
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def transcribe_elevenlabs(audio_wav_bytes: bytes) -> str:
    api_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ELEVENLABS_API_KEY environment variable")

    # Allow overriding the endpoint via env in case of API changes
    url = os.getenv("ELEVEN_STT_URL", "https://api.elevenlabs.io/v1/speech-to-text")
    headers = {"xi-api-key": api_key}

    # ElevenLabs STT expects multipart with field name 'file'
    files = {"file": ("audio.wav", audio_wav_bytes, "audio/wav")}

    # Optional extras: model and language
    data = {}
    model_id = os.getenv("ELEVEN_STT_MODEL")
    if model_id:
        data["model_id"] = model_id
    language = os.getenv("ELEVEN_STT_LANGUAGE")
    if language:
        data["language"] = language

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Surface server response for easier debugging (e.g., 422 with validation message)
        detail = None
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"ElevenLabs STT request failed: {resp.status_code} {detail}") from e
    data = resp.json()
    return data.get("text") or data.get("transcript") or ""


def listen_and_transcribe(seconds: int = 4) -> str:
    audio = record_wav_bytes(seconds=seconds)
    return transcribe_elevenlabs(audio)
