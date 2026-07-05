# capstone — lesson 11

Covers [11 · Composing an Audio AI Pipeline](../../lessons/11-capstone.html).
No new model, no Dockerfile — this is a small host-run script that chains
`foundations`, `speech-to-text`, and `diarization`'s HTTP APIs into one
per-speaker transcript, exactly as lesson 11 describes.

## 1. Bring up the modules this pipeline needs

```
docker compose up -d foundations speech-to-text diarization
```

(`diarization` needs `HF_TOKEN` set — see its README if you haven't done
that setup yet.)

## 2. Run the pipeline

This is the one script in `practices/` that runs on your **host**, not in a
container — it only makes HTTP calls to the containers above, so it just
needs `uv` and `requests`:

```
cd practices/capstone
uv run python pipeline.py ../../data/samples/hello.wav
```

Output:
```
1/3 cleaning audio (foundations: trim/normalize/denoise)...
2/3 transcribing (speech-to-text: Whisper)...
3/3 diarizing (diarization: pyannote)...

Per-speaker transcript:
  [  0.00s -   2.10s] SPEAKER_00: hello, this is a test
```

## Notes

- On a single-speaker clip (like the checked-in sample) this is a trivial
  demo — it gets genuinely interesting on a real multi-person recording,
  where you can see where ASR and diarization timestamps disagree at
  speaker-change boundaries (lesson 11's "errors compound" warning, made
  visible).
- `docker compose down` when you're done — this script doesn't manage the
  containers for you.
