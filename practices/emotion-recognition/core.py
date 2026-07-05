"""
Lesson 08 study code: speech emotion recognition with a fine-tuned wav2vec2.
See lessons/08-emotion-recognition.html for the concepts this demonstrates.

Run directly: python core.py /data/samples/some_clip.wav
"""
import sys

import librosa
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# A public, non-gated wav2vec2 fine-tuned for categorical emotion
# classification on IEMOCAP -- see lesson 08's "acted vs. spontaneous"
# caveat before reading too much into results on real-world audio.
MODEL_NAME = "superb/wav2vec2-base-superb-er"

_CACHE: dict = {}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model_and_extractor():
    if "model" not in _CACHE:
        device = get_device()
        print(f"Loading {MODEL_NAME} on {device}...", file=sys.stderr)
        _CACHE["extractor"] = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        _CACHE["model"] = model
    return _CACHE["model"], _CACHE["extractor"]


def _load_audio_16k_mono(audio_path: str):
    y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y


def predict(audio_path: str) -> dict:
    model, extractor = get_model_and_extractor()
    y = _load_audio_16k_mono(audio_path)
    inputs = extractor(y, sampling_rate=16000, return_tensors="pt")
    device = get_device()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

    labels = model.config.id2label
    scores = {labels[i]: round(float(p), 4) for i, p in enumerate(probs)}
    top_label = max(scores, key=scores.get)
    return {"emotion": top_label, "scores": scores}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core.py <path-to-audio-file>")
        sys.exit(1)

    print(f"Device: {get_device()}")
    result = predict(sys.argv[1])
    print(f"\nPredicted emotion: {result['emotion']}")
    print("Scores:")
    for label, score in sorted(result["scores"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:>10}: {score:.4f}")
