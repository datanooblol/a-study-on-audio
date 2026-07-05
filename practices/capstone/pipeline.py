"""
Lesson 11 study code: composing foundations + speech-to-text + diarization
into one per-speaker transcript.
See lessons/11-capstone.html for the concepts this demonstrates.

Unlike every other module, this one runs on the HOST, not in a container --
it just calls the other modules' HTTP APIs, which must already be up:

  docker compose up -d foundations speech-to-text diarization

Run: uv run python pipeline.py <path-to-audio-file>
"""
import sys
from pathlib import Path

import requests

FOUNDATIONS_URL = "http://localhost:8000"
STT_URL = "http://localhost:8001"
DIARIZATION_URL = "http://localhost:8002"

# All modules share the repo's data/ folder, bind-mounted to /data in every
# container. A module's API returns container-internal paths like
# "/data/foo.wav" -- this maps that back to the real path on the host so
# this (host-run) script can read the file for the next stage.
DATA_DIR_HOST = Path(__file__).resolve().parent.parent.parent / "data"


def _to_host_path(container_path: str) -> Path:
    return DATA_DIR_HOST / Path(container_path).name


def _post_file(url: str, path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            resp = requests.post(url, files={"file": f}, timeout=300)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Couldn't reach {url} -- is that module up? "
            f"(docker compose up -d foundations speech-to-text diarization)"
        ) from exc
    resp.raise_for_status()
    return resp.json()


def clean_audio(path: Path) -> Path:
    result = _post_file(f"{FOUNDATIONS_URL}/clean", path)
    return _to_host_path(result["output_file"])


def transcribe(path: Path) -> dict:
    return _post_file(f"{STT_URL}/transcribe", path)


def diarize(path: Path) -> dict:
    return _post_file(f"{DIARIZATION_URL}/diarize", path)


def speaker_at(time_s: float, diarization_segments: list[dict]) -> str:
    """Which diarized speaker a given timestamp falls into."""
    for seg in diarization_segments:
        if seg["start"] <= time_s < seg["end"]:
            return seg["speaker"]
    return "UNKNOWN"


def merge_transcript_and_speakers(transcript: dict, diarization: dict) -> list[dict]:
    """
    The alignment step lesson 11 calls out as where errors compound: ASR and
    diarization are two independent models with their own timestamps,
    matched here by each ASR segment's midpoint -- a simple approach, not
    the more careful word-level alignment a production system would use.
    """
    merged = []
    for seg in transcript["segments"]:
        midpoint = (seg["start"] + seg["end"]) / 2
        speaker = speaker_at(midpoint, diarization["segments"])
        merged.append({"start": seg["start"], "end": seg["end"], "speaker": speaker, "text": seg["text"]})
    return merged


def run_pipeline(audio_path: str) -> list[dict]:
    original = Path(audio_path)

    print("1/3 cleaning audio (foundations: trim/normalize/denoise)...")
    cleaned = clean_audio(original)

    print("2/3 transcribing (speech-to-text: Whisper)...")
    transcript = transcribe(cleaned)

    print("3/3 diarizing (diarization: pyannote)...")
    diarization = diarize(cleaned)

    return merge_transcript_and_speakers(transcript, diarization)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python pipeline.py <path-to-audio-file>")
        sys.exit(1)

    try:
        merged = run_pipeline(sys.argv[1])
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    print("\nPer-speaker transcript:")
    for seg in merged:
        print(f"  [{seg['start']:>6.2f}s - {seg['end']:>6.2f}s] {seg['speaker']}: {seg['text']}")
