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

1. **Lessons phase (current)** — rich-text HTML lessons covering concepts,
   history, and intuition for each topic, with diagrams. No code yet.
2. **Code phase (later)** — hands-on notebooks/scripts implementing what the
   lessons cover, once the lesson set feels solid.

Don't jump ahead to code or tooling setup until asked — the current goal is
conceptual lessons only.

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

## Working with me on this repo

- When adding a new lesson, match the structure and visual style of existing
  ones — check an existing lesson file before writing a new one from scratch.
- Don't write lesson code samples yet (phase 2). If a concept needs a formula,
  show the formula/intuition, not an implementation.
- This is solo study material — no need for contribution guidelines, tests,
  or CI. Keep the repo lightweight.
