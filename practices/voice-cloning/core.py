"""
Lesson 10 study code: zero-shot voice cloning with Coqui's YourTTS.
See lessons/10-voice-cloning.html for the concepts this demonstrates.

Requires consent from whoever's voice is in the reference clip -- see this
module's README before pointing it at anyone else's recording.

Run directly:
  python core.py "hello, this is a test" /data/samples/reference_voice.wav
"""
import sys

import torch
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"

_CACHE: dict = {}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_tts() -> TTS:
    if "tts" not in _CACHE:
        device = get_device()
        print(f"Loading {MODEL_NAME} on {device}...", file=sys.stderr)
        _CACHE["tts"] = TTS(MODEL_NAME).to(device)
    return _CACHE["tts"]


def clone(text: str, speaker_wav: str, out_path: str, language: str = "en") -> str:
    """
    Zero-shot: no fine-tuning happens here. A speaker embedding is extracted
    from `speaker_wav` and used to condition the TTS model directly -- the
    same embedding-conditioning idea as SV2TTS (lesson 10's history), reusing
    the same family of embeddings as speaker-verification (lesson 07), just
    for generation instead of matching.
    """
    tts = get_tts()
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=out_path)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python core.py "text to speak" <path-to-reference-voice.wav> [output_path] [language]')
        sys.exit(1)

    text = sys.argv[1]
    speaker_wav = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "/data/voice_cloning_output.wav"
    language = sys.argv[4] if len(sys.argv) > 4 else "en"

    print(f"Device: {get_device()}")
    clone(text, speaker_wav, out_path, language)
    print(f"Saved: {out_path}")
