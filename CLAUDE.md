# AI + Audio: A Study Repo

This repo is a personal curriculum for learning how AI is applied to audio —
speech-to-text, text-to-speech, voice cloning, diarization, speaker
verification, emotion detection, and the signal-processing foundations under
all of them.

**Audience**: the user is a data scientist, comfortable with ML/AI concepts in
general (models, training, embeddings, evaluation) but new to audio-specific
signal processing and the audio ML ecosystem. Lessons should build intuition
from that starting point — don't re-explain what a neural network or an
embedding is; do explain what's audio-specific (sampling, spectrograms,
vocoders, DER, WER, MOS, etc.).

## Phases

1. **Lessons phase (done)** — rich-text HTML lessons covering concepts,
   history, and intuition for each topic, with diagrams. `lessons/` is
   considered complete; edit it only for corrections or explicit additions,
   not as part of practices work.
2. **Practices phase (current)** — `practices/`, a Docker + uv sandbox with
   one module per lesson (or bundled group of lessons) pairing a readable
   study script with a thin API, so real models run without native installs
   on the host. See "Practice module conventions" below.

## Repo structure

```
lessons/
  00-curriculum.html       # syllabus / index page, links to every lesson
  01-foundations.html
  02-formats-and-conversion.html
  03-preprocessing.html
  04-feature-extraction.html
  05-speech-to-text.html
  06-diarization.html
  07-speaker-verification.html
  08-emotion-recognition.html
  09-text-to-speech.html
  10-voice-cloning.html
  11-capstone.html
  assets/
    style.css              # shared styles for all lesson pages
```

Lessons are numbered in teaching order (see `lessons/00-curriculum.html` for
the full syllabus and rationale). Keep the numbering when adding lessons;
insert as `NNa-` if something needs to slot between existing numbers rather
than renumbering everything.

```
docker-compose.yml
.env.example                 # copy to .env; HF_TOKEN needed only for diarization
data/                         # shared input/output volume, mounted into every container
  samples/hello.wav           # checked-in real speech sample (generated via
                               # PowerShell's System.Speech.Synthesis), used across
                               # every module's README examples
models_cache/                # per-module model weight cache (gitignored)
practices/
  foundations/                # lessons 01-04, bundled (CPU-only, shared deps)
    Dockerfile
    pyproject.toml            # uv-managed venv (uv sync)
    core_01_sampling.py       # one study script per bundled lesson
    core_02_conversion.py
    core_03_preprocessing.py
    core_04_features.py
    app/main.py                # FastAPI wrapping all four
    README.md
  speech-to-text/              # lesson 05 (GPU)
    Dockerfile / requirements.txt / core.py / app/main.py / README.md
  diarization/                 # lesson 06 (GPU, needs HF_TOKEN)
  speaker-verification/        # lesson 07 (GPU)
  emotion-recognition/         # lesson 08 (GPU)
  text-to-speech/              # lesson 09 (GPU)
  voice-cloning/               # lesson 10 (GPU)
  capstone/                    # lesson 11 -- no Docker, host-run script
    pipeline.py                # calls the other modules' HTTP APIs
    pyproject.toml
    README.md
```

## Lesson page conventions

- Each lesson is a **single self-contained HTML file** (embed or link only to
  `assets/style.css` and `assets/*` — no CDN/external dependencies, so lessons
  work offline and don't rot).
- Support both light and dark viewing: use `prefers-color-scheme` in
  `assets/style.css` rather than hardcoding colors per-page.
- Structure each lesson with these sections (skip any that don't apply, don't
  force all of them):
  1. **Overview** — what this topic is and why it matters, 2-3 sentences.
  2. **Key concepts** — the core ideas, defined precisely.
  3. **How it works** — the pipeline/architecture, with a diagram (inline SVG
     or styled HTML/CSS boxes — no external image dependencies).
  4. **A little history** — how the field got here (e.g. HMM-GMM → neural →
     transformer-era for ASR); helps place today's tools in context.
  5. **Tools & libraries** — what you'd reach for in practice (named, not
     installed/used yet — that's the code phase).
  6. **Evaluation** — how quality is measured for this task (WER, MOS, DER,
     EER, etc.), since a data scientist will want the metric before the model.
  7. **Further reading** — a short list of papers/docs, only if genuinely
     useful.
- Visuals belong inline in the HTML (diagrams, comparison tables, labeled
  waveforms/spectrogram sketches) — this is a visual-first study format, not
  a text dump.
- Keep prose tight. This is study material, not a blog post — prefer
  labeled diagrams and tables over long paragraphs where a picture would
  teach faster.

## Practice module conventions

- Every module pairs two things: a **study script** (`core.py`, or
  `core_0N_name.py` for `foundations`' four bundled lessons) with the actual,
  readable implementation of the lesson's concept, and a **thin FastAPI
  wrapper** (`app/main.py`) that imports those same functions — no logic
  duplicated between the two. The script is what you read; the API is for
  programmatic use and for `capstone` to chain modules together.
- Study scripts are meant to be annotated for learning — comments here
  explain the audio-engineering *why* (e.g. why dithering decorrelates
  quantization error), which is a deliberate exception to writing
  comment-free code elsewhere: these files' whole purpose is pedagogical.
- Each `core*.py` ends with an `if __name__ == "__main__":` block that runs
  a self-contained demo (synthetic audio it generates itself where
  possible, or `data/samples/hello.wav`) and prints/saves real output —
  running it should always produce something concrete to look at, not just
  exercise the code path.
- **Base images**: `foundations` uses `python:3.11-slim` (CPU-only, no heavy
  preinstalled deps). Every other module uses
  `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime` so torch is already
  correctly matched to CUDA/cuDNN — GPU "just works" without separate
  version-matching.
- **uv usage differs by base image, deliberately**: `foundations` is a
  normal uv project (`pyproject.toml` + `uv sync`, run via `uv run python
  ...`). GPU modules install their *extra* packages with
  `uv pip install --system -r requirements.txt` instead — torch is
  deliberately never listed in those requirements files, since a fresh
  `uv sync` would re-download it from PyPI and could land a build that
  doesn't match the base image's CUDA. Run GPU modules' scripts with plain
  `python core.py`, not `uv run` (there's no project venv to run inside).
  If a GPU module's library transitively depends on `torchaudio`
  (speechbrain does), pin it explicitly (e.g. `torchaudio==2.4.1`) to stop
  pip/uv's resolver from silently upgrading torch to satisfy it.
- One container per module, except `foundations` which deliberately bundles
  lessons 01–04 (same lightweight CPU deps, no reason to split). `capstone`
  (lesson 11) is the one exception with no Dockerfile at all — it's a
  host-run script that only makes HTTP calls to the other running
  containers, so it doesn't need containerizing.
- Every module's `README.md` is self-contained: what it does, how to build
  it, how to run the study script directly, how to run the API with an
  example request, and how to tear down — written so picking up one module
  doesn't require reading the others. Follow this shape for new modules.
- Gated/licensed models (pyannote's diarization pipeline) need a manual,
  documented step (accept the license, get a token) — call this out loudly
  in that module's README rather than trying to route around it.
- Before considering a module "done," actually build it and run both the
  script and the API against a real file — see the `foundations` and
  `speech-to-text` READMEs' verification history for the kind of subtle
  bugs this catches (e.g. denoising that quietly regresses SNR at mild
  noise levels, or a normalize-then-denoise ordering that undoes the
  normalization). Not every module needs to be built immediately — it's
  fine to scaffold a module fully and leave it unverified until that lesson
  is revisited, but say so explicitly rather than implying it's been tested.

## Working with me on this repo

- When adding a new lesson, match the structure and visual style of existing
  ones — check an existing lesson file before writing a new one from scratch.
  Treat `lessons/` as done; only touch it for corrections or an explicit
  request for a new lesson.
- When adding a new practice module, match an existing module's file layout
  and README shape (see "Practice module conventions" above) rather than
  inventing a new structure.
- This is solo study/practice material — no need for contribution
  guidelines, tests, or CI. Keep the repo lightweight.
