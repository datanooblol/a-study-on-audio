"""
Lesson 04 study code: hand-crafted acoustic features.
See lessons/04-feature-extraction.html for the concepts this demonstrates.

Learned embeddings (wav2vec2/HuBERT) are deliberately not demonstrated here —
they need torch, which the GPU modules (starting with speech-to-text)
already carry. Keeping this module CPU-only and light stays true to what
lesson 04 calls the "hand-crafted" half of the story.

Run directly: uv run python core_04_features.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

from core_01_sampling import generate_tone


def compute_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = 80) -> np.ndarray:
    """Frequency reshaped onto the mel scale — the default input for most speech/TTS models."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def compute_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Log-mel + DCT -> a compact, decorrelated feature set, the classical ASR/speaker-rec workhorse."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def compute_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Energy folded into the 12 musical pitch classes -- mainly useful for music, not speech."""
    return librosa.feature.chroma_stft(y=y, sr=sr)


def compute_pitch_contour(y: np.ndarray, sr: int) -> np.ndarray:
    """Fundamental frequency (F0) over time via the pYIN algorithm -- central to prosody/emotion/TTS."""
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_clean = np.where(voiced_flag, f0, np.nan)
    return f0_clean


def save_feature_overview_plot(y: np.ndarray, sr: int, out_path: str) -> None:
    mel_db = compute_mel_spectrogram(y, sr)
    mfcc = compute_mfcc(y, sr)
    chroma = compute_chroma(y, sr)
    f0 = compute_pitch_contour(y, sr)

    fig, axes = plt.subplots(4, 1, figsize=(8, 11))

    img0 = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[0])
    axes[0].set(title="Mel-spectrogram")
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

    img1 = librosa.display.specshow(mfcc, x_axis="time", ax=axes[1])
    axes[1].set(title="MFCCs", ylabel="coefficient")
    fig.colorbar(img1, ax=axes[1])

    img2 = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=axes[2])
    axes[2].set(title="Chroma")
    fig.colorbar(img2, ax=axes[2])

    times = librosa.times_like(f0, sr=sr)
    axes[3].plot(times, f0)
    axes[3].set(title="Pitch contour (F0, pYIN)", xlabel="time (s)", ylabel="Hz")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sr = 16000
    # A gliding tone (rising pitch) makes the pitch-contour plot obviously meaningful.
    duration = 2.0
    t = np.arange(int(sr * duration)) / sr
    glide = np.sin(2 * np.pi * (220 + 220 * t) * t).astype(np.float32)

    mel_db = compute_mel_spectrogram(glide, sr)
    mfcc = compute_mfcc(glide, sr)
    chroma = compute_chroma(glide, sr)
    f0 = compute_pitch_contour(glide, sr)

    print(f"Mel-spectrogram shape: {mel_db.shape} (n_mels x time_frames)")
    print(f"MFCC shape:            {mfcc.shape} (n_mfcc x time_frames)")
    print(f"Chroma shape:          {chroma.shape} (12 pitch classes x time_frames)")
    print(f"Pitch contour:         {f0.shape[0]} frames, "
          f"range {np.nanmin(f0):.0f}-{np.nanmax(f0):.0f} Hz")

    print("\nSaving feature overview plot to /data/foundations_04_features.png")
    save_feature_overview_plot(glide, sr, "/data/foundations_04_features.png")
    print("Done.")
