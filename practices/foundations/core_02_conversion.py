"""
Lesson 02 study code: safe vs. naive format conversion.
See lessons/02-formats-and-conversion.html for the concepts this demonstrates.

Run directly: uv run python core_02_conversion.py
"""
import numpy as np
import librosa

from core_01_sampling import generate_tone, dominant_frequency


def naive_downsample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    The WRONG way to change sample rate: just drop samples, with no
    anti-aliasing filter first. This is the mistake lesson 02 warns about.
    """
    step = orig_sr // target_sr
    return y[::step]


def safe_resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    The correct way: librosa's resampler applies a proper band-limiting
    filter before changing the sample rate, so content above the new
    Nyquist frequency is removed instead of folding back as aliasing.
    """
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def compare_resampling_methods(true_freq_hz: float, orig_sr: int, target_sr: int) -> dict:
    """Downsample the same above-Nyquist tone both ways and compare what each thinks the frequency is."""
    y = generate_tone(true_freq_hz, 1.0, orig_sr)
    naive = naive_downsample(y, orig_sr, target_sr)
    safe = safe_resample(y, orig_sr, target_sr)
    return {
        "true_freq_hz": true_freq_hz,
        "target_nyquist_hz": target_sr / 2,
        "naive_observed_hz": round(dominant_frequency(naive, target_sr), 1),
        "safe_observed_hz": round(dominant_frequency(safe, target_sr), 1),
    }


def quantize(y_float: np.ndarray, bit_depth: int, dither: bool) -> np.ndarray:
    """
    Reduce bit depth by scaling to an integer range and rounding.
    With dither=True, a small amount of triangular noise is added *before*
    rounding, decorrelating the resulting quantization error from the
    signal (see lesson 02's dithering explanation).
    """
    max_val = 2 ** (bit_depth - 1) - 1
    scaled = y_float * max_val
    if dither:
        # Triangular dither: sum of two uniform distributions, the
        # standard choice for audio dithering.
        noise = (np.random.uniform(-0.5, 0.5, size=y_float.shape) + np.random.uniform(-0.5, 0.5, size=y_float.shape))
        scaled = scaled + noise
    quantized = np.round(scaled)
    return (quantized / max_val).astype(np.float32)


def quantization_error_stats(y_float: np.ndarray, bit_depth: int) -> dict:
    """Compare the quantization error's correlation with the signal, with vs. without dither."""
    no_dither = quantize(y_float, bit_depth, dither=False)
    with_dither = quantize(y_float, bit_depth, dither=True)
    err_no_dither = y_float - no_dither
    err_with_dither = y_float - with_dither
    # Correlation close to 0 is good — it means the error doesn't track the
    # signal (i.e. doesn't sound like structured distortion).
    corr_no_dither = float(np.corrcoef(y_float, err_no_dither)[0, 1])
    corr_with_dither = float(np.corrcoef(y_float, err_with_dither)[0, 1])
    return {
        "bit_depth": bit_depth,
        "error_signal_correlation_no_dither": round(corr_no_dither, 4),
        "error_signal_correlation_with_dither": round(corr_with_dither, 4),
    }


def downmix_stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Simple stereo-to-mono averaging — fine unless the channels are out of phase (see below)."""
    return (left + right) / 2.0


def phase_cancellation_demo(freq_hz: float, sample_rate: int, duration_s: float = 1.0) -> dict:
    """
    Worst case for naive downmixing: right channel is the exact inverse of
    left. Averaging them cancels the shared frequency almost entirely.
    """
    left = generate_tone(freq_hz, duration_s, sample_rate)
    right = -left  # fully out of phase
    mono = downmix_stereo(left, right)
    return {
        "left_rms": float(np.sqrt(np.mean(left**2))),
        "right_rms": float(np.sqrt(np.mean(right**2))),
        "downmixed_rms": float(np.sqrt(np.mean(mono**2))),
    }


if __name__ == "__main__":
    print("Resampling an 18000 Hz tone down to 16000 Hz sample rate (Nyquist becomes 8000 Hz):")
    result = compare_resampling_methods(true_freq_hz=18000, orig_sr=44100, target_sr=16000)
    print(f"  Naive (no filter): observed {result['naive_observed_hz']} Hz -- aliased, wrong")
    print(f"  Safe  (filtered):  observed {result['safe_observed_hz']} Hz -- correctly attenuated/removed")

    print("\nBit-depth reduction to 8-bit: does dithering decorrelate the error from the signal?")
    sr = 16000
    tone = generate_tone(440, 1.0, sr)
    stats = quantization_error_stats(tone, bit_depth=8)
    print(f"  Error/signal correlation without dither: {stats['error_signal_correlation_no_dither']}")
    print(f"  Error/signal correlation with dither:    {stats['error_signal_correlation_with_dither']} (closer to 0 is better)")

    print("\nStereo->mono downmix with fully out-of-phase channels (phase cancellation):")
    cancel = phase_cancellation_demo(440, sr)
    print(f"  Left RMS: {cancel['left_rms']:.3f}, Right RMS: {cancel['right_rms']:.3f}, "
          f"Downmixed RMS: {cancel['downmixed_rms']:.6f} (should be ~0 -- the content vanished)")
