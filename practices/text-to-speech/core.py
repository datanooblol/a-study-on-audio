"""
Lesson 09 study code: text-to-speech with Coqui TTS (a VITS model, lesson
09's end-to-end acoustic-model + vocoder architecture, trained on VCTK).
See lessons/09-text-to-speech.html for the concepts this demonstrates.

Run directly: python core.py "hello, this is a test" /data/tts_output.wav
"""
import sys

import torch
from TTS.api import TTS

MODEL_NAME = "tts_models/en/vctk/vits"

_CACHE: dict = {}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_tts() -> TTS:
    if "tts" not in _CACHE:
        device = get_device()
        print(f"Loading {MODEL_NAME} on {device}...", file=sys.stderr)
        _CACHE["tts"] = TTS(MODEL_NAME).to(device)
    return _CACHE["tts"]


def list_speakers() -> list[str]:
    """VCTK is a multi-speaker dataset -- this model can render any of ~100 built-in voices."""
    tts = get_tts()
    return tts.speakers or []


def synthesize(text: str, out_path: str, speaker: str | None = None) -> str:
    tts = get_tts()
    kwargs = {}
    if tts.is_multi_speaker:
        kwargs["speaker"] = speaker or tts.speakers[0]
    tts.tts_to_file(text=text, file_path=out_path, **kwargs)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python core.py "text to speak" [output_path] [speaker_id]')
        sys.exit(1)

    text = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/data/text_to_speech_output.wav"
    speaker = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Device: {get_device()}")
    print(f"Available speakers (first 5): {list_speakers()[:5]}")
    synthesize(text, out_path, speaker)
    print(f"Saved: {out_path}")
