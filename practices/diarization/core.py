"""
Lesson 06 study code: speaker diarization with pyannote.audio.
See lessons/06-diarization.html for the concepts this demonstrates.

Requires HF_TOKEN (see README.md for how to get one).

Run directly: python core.py /data/samples/some_meeting.wav
"""
import os
import sys

import torch
from pyannote.audio import Pipeline

_PIPELINE_CACHE: dict[str, Pipeline] = {}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_pipeline() -> Pipeline:
    """
    The VAD -> embed -> cluster pipeline from lesson 06, packaged as one
    pretrained pipeline. Gated on Hugging Face -- HF_TOKEN must have accepted
    the model's license (README.md walks through this).
    """
    if "pipeline" not in _PIPELINE_CACHE:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set. See practices/diarization/README.md for how to "
                "create one and accept the pyannote/speaker-diarization-3.1 license."
            )
        print("Loading pyannote/speaker-diarization-3.1...", file=sys.stderr)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        pipeline.to(torch.device(get_device()))
        _PIPELINE_CACHE["pipeline"] = pipeline
    return _PIPELINE_CACHE["pipeline"]


def diarize(audio_path: str) -> dict:
    """Runs the full pipeline and flattens its output into a simple segment list."""
    pipeline = get_pipeline()
    annotation = pipeline(audio_path)
    segments = [
        {"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    return {"segments": segments, "num_speakers": len({s["speaker"] for s in segments})}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core.py <path-to-audio-file>")
        sys.exit(1)

    print(f"Device: {get_device()}")
    result = diarize(sys.argv[1])
    print(f"\nDetected {result['num_speakers']} speaker(s):")
    for seg in result["segments"]:
        print(f"  [{seg['start']:>6.2f}s - {seg['end']:>6.2f}s] {seg['speaker']}")
