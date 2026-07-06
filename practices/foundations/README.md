# foundations — lessons 01–04

Covers [01 · Foundations of Digital Audio](../../lessons/01-foundations.html),
[02 · Audio Formats & Conversion](../../lessons/02-formats-and-conversion.html),
[03 · Audio Preprocessing & Manipulation](../../lessons/03-preprocessing.html),
and [04 · Feature Extraction & Audio Embeddings](../../lessons/04-feature-extraction.html).
CPU-only — no GPU needed.

| File | Lesson | What it demonstrates |
|---|---|---|
| `core_01_sampling.py` | 01 | Bit-depth/dynamic-range table, a live Nyquist aliasing demo, waveform + spectrogram plot |
| `core_02_conversion.py` | 02 | Naive vs. safe resampling (aliasing side by side), dithering, stereo phase-cancellation |
| `core_03_preprocessing.py` | 03 | Silence trimming, LUFS loudness normalization, denoising + SNR |
| `core_04_features.py` | 04 | Mel-spectrogram, MFCC, chroma, pitch contour |

## 1. Study the code directly (no server, no file upload needed)

`core_01_sampling.py`, `core_02_conversion.py`, and `core_04_features.py`
run self-contained demos against synthetic audio they generate themselves —
no file needed. `core_03_preprocessing.py` uses the checked-in real speech
sample at `data/samples/hello.wav` (denoising is much more meaningful on
real speech than a synthetic tone — falls back to a synthetic tone if that
file is ever missing).

```
docker compose build foundations
docker compose run --rm foundations uv run python core_01_sampling.py
docker compose run --rm foundations uv run python core_02_conversion.py
docker compose run --rm foundations uv run python core_03_preprocessing.py
docker compose run --rm foundations uv run python core_04_features.py
```

Plots get written to `/data` inside the container, which is the repo's
`data/` folder on your host — open them there after running.

## 2. Run the API, against your own audio files

```
docker compose up -d foundations
curl http://localhost:8000/health
```

Put a file you want to experiment with into `data/` first (or use one of the
samples in `data/samples/`), then:

```
# Lesson 01 demo — no file needed
curl -X POST http://localhost:8000/sampling/demo

# Lesson 02 — convert a file's sample rate, naively or safely
curl -X POST http://localhost:8000/convert \
  -F "file=@data/samples/hello.wav" \
  -F "target_sample_rate=8000" \
  -F "method=naive"

# Lesson 03 — trim, denoise, then normalize loudness (in that order — see the lesson)
curl -X POST http://localhost:8000/clean -F "file=@data/samples/hello.wav"

# Lesson 04 — extract features
curl -X POST http://localhost:8000/features -F "file=@data/samples/hello.wav"
```

Every endpoint writes its output (a `.wav` or a `.png`) into `/data`
(`./data` on the host) and returns its path in the JSON response.

```
docker compose down
```
