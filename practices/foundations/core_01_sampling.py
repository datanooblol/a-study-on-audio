"""
Lesson 01 study code: sampling, bit depth, and the time/frequency duality.
See lessons/01-foundations.html for the concepts this demonstrates.

Run directly: uv run python core_01_sampling.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display inside a container
import matplotlib.pyplot as plt
import librosa
import librosa.display


def bit_depth_table() -> list[dict]:
    """Values-per-sample and dynamic range for common bit depths.

    Dynamic range in dB follows the standard ~6.02 dB/bit rule
    (20*log10(2) per extra bit), plus the 1.76 dB constant from
    full-scale sine quantization theory.
    """
    depths = [8, 16, 24, 32]
    table = []
    for bits in depths:
        values = 2**bits
        dynamic_range_db = 6.02 * bits + 1.76
        table.append({"bit_depth": bits, "values_per_sample": values, "dynamic_range_db": round(dynamic_range_db, 1)})
    return table


def generate_tone(freq_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    """A pure sine tone — the simplest possible signal to reason about."""
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def dominant_frequency(signal: np.ndarray, sample_rate: int) -> float:
    """The frequency with the most energy, via FFT — 'what does this signal sound like'."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
    return float(freqs[np.argmax(np.abs(spectrum))])


def demonstrate_aliasing(true_freq_hz: float, sample_rate: int, duration_s: float = 1.0) -> dict:
    """
    The Nyquist theorem in one function: sample a tone above sample_rate/2
    and show the FFT reports a *different, lower* frequency instead — the
    signal has been aliased, and there is no way to tell from the samples
    alone that this happened.
    """
    nyquist = sample_rate / 2
    sampled = generate_tone(true_freq_hz, duration_s, sample_rate)
    observed_freq = dominant_frequency(sampled, sample_rate)
    return {
        "true_freq_hz": true_freq_hz,
        "sample_rate": sample_rate,
        "nyquist_hz": nyquist,
        "violates_nyquist": true_freq_hz > nyquist,
        "observed_freq_hz": round(observed_freq, 1),
    }


def save_waveform_and_spectrogram_plot(y: np.ndarray, sr: int, out_path: str, title: str) -> None:
    """The two views from the lesson, side by side: amplitude-over-time vs. frequency-over-time."""
    stft = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(np.arange(len(y)) / sr, y, linewidth=0.8)
    axes[0].set(title=f"{title} — waveform", xlabel="time (s)", ylabel="amplitude")

    img = librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set(title=f"{title} — spectrogram (STFT)")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    print("Bit depth / dynamic range:")
    for row in bit_depth_table():
        print(f"  {row['bit_depth']:>2}-bit: {row['values_per_sample']:>12,} values, ~{row['dynamic_range_db']} dB dynamic range")

    print("\nNyquist / aliasing demo:")
    well_sampled = demonstrate_aliasing(true_freq_hz=1000, sample_rate=8000)
    aliased = demonstrate_aliasing(true_freq_hz=9000, sample_rate=8000)
    print(f"  1000 Hz tone @ 8000 Hz sample rate -> observed {well_sampled['observed_freq_hz']} Hz (correct, below Nyquist)")
    print(f"  9000 Hz tone @ 8000 Hz sample rate -> observed {aliased['observed_freq_hz']} Hz "
          f"(WRONG — aliased, since 9000 Hz > {aliased['nyquist_hz']} Hz Nyquist limit)")

    print("\nSaving waveform + spectrogram plot for a 3-tone chord to /data/foundations_01_waveform_spectrogram.png")
    sr = 16000
    chord = (
        generate_tone(220, 2.0, sr) * 0.5
        + generate_tone(440, 2.0, sr) * 0.3
        + generate_tone(880, 2.0, sr) * 0.2
    ).astype(np.float32)
    save_waveform_and_spectrogram_plot(chord, sr, "/data/foundations_01_waveform_spectrogram.png", "220/440/880 Hz chord")
    print("Done.")
