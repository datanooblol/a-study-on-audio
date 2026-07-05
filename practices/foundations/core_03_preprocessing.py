"""
Lesson 03 study code: trimming, loudness normalization, and denoising.
See lessons/03-preprocessing.html for the concepts this demonstrates.

Run directly: uv run python core_03_preprocessing.py
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import noisereduce as nr

from core_01_sampling import generate_tone


def add_silence_padding(y: np.ndarray, sr: int, pad_s: float = 0.5) -> np.ndarray:
    """Simulate a raw recording with dead air at both ends."""
    pad = np.zeros(int(sr * pad_s), dtype=y.dtype)
    return np.concatenate([pad, y, pad])


def trim_silence(y: np.ndarray, top_db: float = 30.0) -> tuple[np.ndarray, tuple[int, int]]:
    """Energy-based VAD: drop leading/trailing regions quieter than top_db below peak."""
    trimmed, index = librosa.effects.trim(y, top_db=top_db)
    return trimmed, (int(index[0]), int(index[1]))


def measure_loudness_lufs(y: np.ndarray, sr: int) -> float:
    """
    Integrated loudness in LUFS (ITU-R BS.1770), the perceptual measure
    lesson 03 recommends over simple peak normalization.
    """
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(y.astype(np.float64)))


def normalize_loudness(y: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """Scale a signal so its integrated loudness matches a target (broadcast standard is -23 LUFS)."""
    current = measure_loudness_lufs(y, sr)
    return pyln.normalize.loudness(y.astype(np.float64), current, target_lufs).astype(np.float32)


def denoise(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Spectral-gating noise reduction (see noisereduce's approach). Worth
    knowing from running this demo: denoising isn't free -- its own
    residual/artifact floor means it only shows a clear SNR win once the
    input is genuinely noisy. On already-mild noise it can look *worse*
    than doing nothing, because the algorithm's leftover artifacts outweigh
    the small amount of noise it removed.
    """
    return nr.reduce_noise(y=y, sr=sr, stationary=True)


def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Signal-to-noise ratio in dB — lesson 03's primary evaluation metric."""
    noise = noisy[: len(clean)] - clean
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2) + 1e-12
    return float(10 * np.log10(signal_power / noise_power))


if __name__ == "__main__":
    sample_path = Path("/data/samples/hello.wav")
    if sample_path.exists():
        print(f"Using real speech sample: {sample_path}")
        clean_tone, sr = sf.read(sample_path, dtype="float32")
    else:
        print("No sample file found at /data/samples/hello.wav -- using a synthetic tone instead.")
        sr = 16000
        clean_tone = generate_tone(440, 2.0, sr) * 0.4

    rng = np.random.default_rng(0)
    # Loud enough that denoising actually has a noise floor worth removing
    # (see the caveat in denoise()'s docstring above).
    noise = rng.normal(0, 0.1, size=clean_tone.shape).astype(np.float32)
    noisy_tone = clean_tone + noise

    print("Trimming silence:")
    padded = add_silence_padding(noisy_tone, sr, pad_s=0.5)
    trimmed, (start, end) = trim_silence(padded, top_db=20)
    print(f"  Padded length: {len(padded)} samples ({len(padded)/sr:.2f}s)")
    print(f"  Trimmed length: {len(trimmed)} samples ({len(trimmed)/sr:.2f}s), kept samples [{start}:{end}]")

    print("\nLoudness normalization (LUFS):")
    quiet = clean_tone * 0.1
    before = measure_loudness_lufs(quiet, sr)
    normalized = normalize_loudness(quiet, sr, target_lufs=-23.0)
    after = measure_loudness_lufs(normalized, sr)
    print(f"  Before: {before:.1f} LUFS -> After: {after:.1f} LUFS (target -23.0)")

    print("\nDenoising + SNR:")
    before_snr = snr_db(clean_tone, noisy_tone)
    denoised = denoise(noisy_tone, sr)
    after_snr = snr_db(clean_tone, denoised)
    print(f"  SNR before denoising: {before_snr:.1f} dB")
    print(f"  SNR after denoising:  {after_snr:.1f} dB")
