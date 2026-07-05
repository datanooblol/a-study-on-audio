"""
Lesson 05 study code: Whisper ASR.
See lessons/05-speech-to-text.html for the concepts this demonstrates.

Run directly: python core.py /data/samples/some_file.wav
"""
import os
import sys

import torch
import whisper

_MODEL_CACHE: dict[str, "whisper.Whisper"] = {}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_size: str | None = None) -> "whisper.Whisper":
    """
    Whisper ships several sizes (tiny/base/small/medium/large) trading speed
    for accuracy. Loaded once and cached -- this is the expensive step, not
    the actual transcription of a single short clip.
    """
    model_size = model_size or os.environ.get("WHISPER_MODEL", "base")
    if model_size not in _MODEL_CACHE:
        device = get_device()
        print(f"Loading whisper '{model_size}' on {device}...", file=sys.stderr)
        _MODEL_CACHE[model_size] = whisper.load_model(model_size, device=device)
    return _MODEL_CACHE[model_size]


def transcribe(audio_path: str, model_size: str | None = None) -> dict:
    """
    Runs Whisper's encoder-decoder end to end: audio in, text + per-segment
    timestamps out (lesson 05's "How it works" diagram, executed for real).
    """
    model = get_model(model_size)
    result = model.transcribe(audio_path)
    segments = [
        {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
        for s in result["segments"]
    ]
    return {"text": result["text"].strip(), "language": result.get("language"), "segments": segments}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core.py <path-to-audio-file> [model_size]")
        print("       (mount a file into /data first, e.g. data/samples/hello.wav on the host)")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Device: {get_device()}")
    result = transcribe(audio_path, model_size)
    print(f"\nDetected language: {result['language']}")
    print(f"Transcript: {result['text']}\n")
    print("Segments:")
    for seg in result["segments"]:
        print(f"  [{seg['start']:>6.2f}s - {seg['end']:>6.2f}s] {seg['text']}")
