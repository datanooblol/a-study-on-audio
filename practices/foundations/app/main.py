"""
Thin FastAPI wrapper around core_01-04's functions. All the actual logic
lives in those files — this just moves bytes in and out over HTTP.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

# core_0*.py live one directory up (see the repo layout in README.md) —
# add it to sys.path so both `uv run python core_01_sampling.py` and
# `uv run uvicorn app.main:app` resolve the same imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core_01_sampling as s01
import core_02_conversion as s02
import core_03_preprocessing as s03
import core_04_features as s04

app = FastAPI(title="foundations — lessons 01-04")
DATA_DIR = Path("/data")


def _load_audio(raw: bytes) -> tuple[np.ndarray, int]:
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)  # downmix to mono for these demos
    return y, sr


def _save_audio(y: np.ndarray, sr: int, name: str) -> str:
    out_path = DATA_DIR / name
    sf.write(out_path, y, sr)
    return str(out_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sampling/demo")
def sampling_demo():
    """Lesson 01: bit-depth table + a live Nyquist/aliasing demonstration."""
    well_sampled = s01.demonstrate_aliasing(true_freq_hz=1000, sample_rate=8000)
    aliased = s01.demonstrate_aliasing(true_freq_hz=9000, sample_rate=8000)

    sr = 16000
    chord = (
        s01.generate_tone(220, 2.0, sr) * 0.5
        + s01.generate_tone(440, 2.0, sr) * 0.3
        + s01.generate_tone(880, 2.0, sr) * 0.2
    ).astype(np.float32)
    plot_path = str(DATA_DIR / "foundations_01_waveform_spectrogram.png")
    s01.save_waveform_and_spectrogram_plot(chord, sr, plot_path, "220/440/880 Hz chord")

    return {
        "bit_depth_table": s01.bit_depth_table(),
        "well_sampled_example": well_sampled,
        "aliased_example": aliased,
        "waveform_spectrogram_plot": plot_path,
    }


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    target_sample_rate: int = Form(16000),
    method: str = Form("safe"),
):
    """Lesson 02: resample an uploaded file, naively or safely, and report what happened."""
    raw = await file.read()
    y, sr = _load_audio(raw)

    if method == "naive":
        converted = s02.naive_downsample(y, sr, target_sample_rate) if target_sample_rate < sr else y
    else:
        converted = s02.safe_resample(y, sr, target_sample_rate)

    out_name = f"foundations_02_converted_{method}_{int(time.time())}.wav"
    out_path = _save_audio(converted, target_sample_rate, out_name)

    return JSONResponse({
        "method": method,
        "original_sample_rate": sr,
        "target_sample_rate": target_sample_rate,
        "original_duration_s": round(len(y) / sr, 3),
        "converted_duration_s": round(len(converted) / target_sample_rate, 3),
        "output_file": out_path,
    })


@app.post("/clean")
async def clean(file: UploadFile = File(...)):
    """
    Lesson 03: trim silence, denoise, then normalize loudness (LUFS) last.
    Denoising changes a signal's energy, so normalizing loudness *before*
    denoising would just get undone by it -- normalizing last is what
    makes the reported "after" loudness actually land on the target.
    """
    raw = await file.read()
    y, sr = _load_audio(raw)

    before_lufs = s03.measure_loudness_lufs(y, sr)
    trimmed, (start, end) = s03.trim_silence(y)
    denoised = s03.denoise(trimmed, sr)
    normalized = s03.normalize_loudness(denoised, sr, target_lufs=-23.0)
    after_lufs = s03.measure_loudness_lufs(normalized, sr)

    out_name = f"foundations_03_cleaned_{int(time.time())}.wav"
    out_path = _save_audio(normalized, sr, out_name)

    return JSONResponse({
        "original_duration_s": round(len(y) / sr, 3),
        "trimmed_duration_s": round(len(trimmed) / sr, 3),
        "loudness_before_lufs": round(before_lufs, 1),
        "loudness_after_lufs": round(after_lufs, 1),
        "output_file": out_path,
    })


@app.post("/features")
async def features(file: UploadFile = File(...)):
    """Lesson 04: compute mel-spectrogram/MFCC/chroma/pitch for an uploaded file."""
    raw = await file.read()
    y, sr = _load_audio(raw)

    mel_db = s04.compute_mel_spectrogram(y, sr)
    mfcc = s04.compute_mfcc(y, sr)
    chroma = s04.compute_chroma(y, sr)
    f0 = s04.compute_pitch_contour(y, sr)

    out_name = f"foundations_04_features_{int(time.time())}.png"
    plot_path = str(DATA_DIR / out_name)
    s04.save_feature_overview_plot(y, sr, plot_path)

    voiced = f0[~np.isnan(f0)]
    return JSONResponse({
        "mel_spectrogram_shape": list(mel_db.shape),
        "mfcc_shape": list(mfcc.shape),
        "chroma_shape": list(chroma.shape),
        "pitch_frames": int(f0.shape[0]),
        "pitch_range_hz": [float(voiced.min()), float(voiced.max())] if len(voiced) else None,
        "plot_file": plot_path,
    })
