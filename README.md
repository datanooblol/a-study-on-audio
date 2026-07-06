# Practice sandbox

Hands-on companion to the lessons in [`lessons/`](lessons/00-curriculum.html).
Each module in [`practices/`](practices/) pairs a **readable study script**
(the actual implementation of that lesson's concept, meant to be opened and
read) with a **thin API** on top of it, both running in a container so none
of the native installs (ffmpeg, libsndfile, espeak-ng, matching CUDA/cuDNN
versions) have to happen on your host — identical on Windows and Linux.

## Prerequisites

- **Docker Desktop** (Windows: with WSL2 backend) or **Docker Engine +
  nvidia-container-toolkit** (Linux).
- A GPU is only required for the six non-`foundations` modules. Verify GPU
  passthrough works before building anything:
  ```
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
  You should see your GPU listed. If this fails, fix that first — none of
  the module containers will work without it.
- `uv` is only needed on your host if you want to run a module's script
  *outside* Docker; it's already installed inside every container.

## Setup

```
cp .env.example .env
```
Fill in `HF_TOKEN` only if you plan to use `diarization` (see its README for
how to get one — everything else works without it).

## Modules

| # | Lesson | Module | Port | Status | What it does |
|---|---|---|---|---|---|
| 01–04 | [Foundations](lessons/01-foundations.html) → [Feature Extraction](lessons/04-feature-extraction.html) | [`foundations`](practices/foundations/README.md) | 8000 | ✅ verified | Sampling/spectrogram demo, format conversion (safe vs. naive), preprocessing, feature extraction |
| 05 | [Speech-to-Text](lessons/05-speech-to-text.html) | [`speech-to-text`](practices/speech-to-text/README.md) | 8001 | ✅ verified | Whisper transcription |
| 06 | [Diarization](lessons/06-diarization.html) | [`diarization`](practices/diarization/README.md) | 8002 | scaffolded, unbuilt | pyannote.audio "who spoke when" |
| 07 | [Speaker Verification](lessons/07-speaker-verification.html) | [`speaker-verification`](practices/speaker-verification/README.md) | 8003 | scaffolded, unbuilt | ECAPA-TDNN enroll/verify/identify |
| 08 | [Emotion Recognition](lessons/08-emotion-recognition.html) | [`emotion-recognition`](practices/emotion-recognition/README.md) | 8004 | scaffolded, unbuilt | wav2vec2 SER |
| 09 | [Text-to-Speech](lessons/09-text-to-speech.html) | [`text-to-speech`](practices/text-to-speech/README.md) | 8005 | scaffolded, unbuilt | Coqui VITS synthesis |
| 10 | [Voice Cloning](lessons/10-voice-cloning.html) | [`voice-cloning`](practices/voice-cloning/README.md) | 8006 | scaffolded, unbuilt | Coqui YourTTS zero-shot cloning |
| 11 | [Capstone](lessons/11-capstone.html) | [`capstone`](practices/capstone/README.md) | — | scaffolded, unbuilt | Host-run script chaining the modules above over HTTP |

Each module's own README has the full step-by-step (build, run the script to
study it, run the API, example request) — this table is just the index.

"✅ verified" means the module has actually been built and run end-to-end
(both the study script and the API) against a real file. "Scaffolded,
unbuilt" means the code is complete and follows the same conventions, but
hasn't been run yet — treat the first build as a first attempt, not a sure
thing, and expect to iron out the odd dependency/version issue the way
`foundations` and `speech-to-text` already had two rounds of fixes each.

## Running a module

Bring modules up **one at a time**, not all at once — running all six GPU
modules simultaneously can exceed a 16GB GPU's VRAM.

```
docker compose build foundations
docker compose up -d foundations
curl http://localhost:8000/health

docker compose down
```

To read and run a module's study script directly instead of the API:
```
docker compose run --rm foundations uv run python core_01_sampling.py
```

> **Git Bash on Windows:** if a command passes an absolute container path
> like `/data/samples/hello.wav` as an argument (not through `curl -F`,
> which is fine), Git Bash's MSYS layer silently rewrites it into a Windows
> path before Docker ever sees it, producing a confusing `ffmpeg`/"Protocol
> not found" error. Fix: prefix the command with `MSYS_NO_PATHCONV=1`, e.g.
> `MSYS_NO_PATHCONV=1 docker compose run --rm speech-to-text python core.py /data/samples/hello.wav`.
> PowerShell and cmd.exe aren't affected.

## Repo structure

```
lessons/       # phase 1 — conceptual HTML lessons (untouched by this work)
practices/     # phase 2 — this sandbox, one folder per module
data/          # shared input/output volume, mounted into every container
models_cache/  # per-module model weight cache (gitignored, created on first run)
docker-compose.yml
.env.example
```
