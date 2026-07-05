"""
Lesson 07 study code: speaker verification & identification with an
ECAPA-TDNN embedding model (speechbrain).
See lessons/07-speaker-verification.html for the concepts this demonstrates.

Voiceprints are enrolled into a plain JSON file under /data — good enough
for a sandbox, not how you'd store them in production.

Run directly:
  python core.py enroll alice /data/samples/alice_1.wav
  python core.py verify alice /data/samples/alice_2.wav
  python core.py identify /data/samples/someone.wav
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
from speechbrain.inference.speaker import EncoderClassifier

VOICEPRINT_STORE = Path("/data/voiceprints.json")
# Cosine similarity threshold for accept/reject -- a rough default (lesson
# 07's EER is the principled way to pick this; this is a starting point to
# tune against your own enrolled voices).
DEFAULT_THRESHOLD = 0.25

_MODEL_CACHE: dict[str, EncoderClassifier] = {}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> EncoderClassifier:
    if "model" not in _MODEL_CACHE:
        device = get_device()
        print(f"Loading speechbrain ECAPA-TDNN on {device}...", file=sys.stderr)
        _MODEL_CACHE["model"] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/root/.cache/speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
    return _MODEL_CACHE["model"]


def _load_audio_16k_mono(audio_path: str) -> torch.Tensor:
    y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return torch.from_numpy(y).unsqueeze(0)


def embed(audio_path: str) -> list[float]:
    """The fixed-length speaker embedding — same idea whether it feeds verification (here) or diarization (lesson 06)."""
    model = get_model()
    signal = _load_audio_16k_mono(audio_path)
    embedding = model.encode_batch(signal)
    return embedding.squeeze().detach().cpu().numpy().tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def _load_store() -> dict:
    if VOICEPRINT_STORE.exists():
        return json.loads(VOICEPRINT_STORE.read_text())
    return {}


def _save_store(store: dict) -> None:
    VOICEPRINT_STORE.write_text(json.dumps(store))


def enroll(speaker_name: str, audio_path: str) -> dict:
    vec = embed(audio_path)
    store = _load_store()
    store[speaker_name] = vec
    _save_store(store)
    return {"speaker": speaker_name, "enrolled": True, "embedding_dim": len(vec)}


def verify(speaker_name: str, audio_path: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    store = _load_store()
    if speaker_name not in store:
        raise ValueError(f"Speaker '{speaker_name}' is not enrolled yet -- run `enroll` first.")
    score = cosine_similarity(embed(audio_path), store[speaker_name])
    return {"speaker": speaker_name, "score": round(score, 4), "match": score >= threshold, "threshold": threshold}


def identify(audio_path: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    store = _load_store()
    if not store:
        raise ValueError("No speakers enrolled yet -- run `enroll` first.")
    vec = embed(audio_path)
    scores = {name: round(cosine_similarity(vec, ref), 4) for name, ref in store.items()}
    best_name = max(scores, key=scores.get)
    return {
        "best_match": best_name if scores[best_name] >= threshold else None,
        "scores": scores,
        "threshold": threshold,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll")
    p_enroll.add_argument("speaker_name")
    p_enroll.add_argument("audio_path")

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("speaker_name")
    p_verify.add_argument("audio_path")

    p_identify = sub.add_parser("identify")
    p_identify.add_argument("audio_path")

    args = parser.parse_args()
    print(f"Device: {get_device()}")

    if args.command == "enroll":
        print(enroll(args.speaker_name, args.audio_path))
    elif args.command == "verify":
        print(verify(args.speaker_name, args.audio_path))
    elif args.command == "identify":
        print(identify(args.audio_path))
