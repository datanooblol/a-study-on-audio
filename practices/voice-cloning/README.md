# voice-cloning — lesson 10

Covers [10 · Voice Cloning](../../lessons/10-voice-cloning.html) using Coqui's
YourTTS for zero-shot cloning from a few seconds of reference audio.
GPU-capable.

> **Only point this at voices you have consent to clone** — your own
> recordings, or audio from someone who's explicitly agreed. Lesson 10 covers
> why this matters (impersonation, fraud, non-consensual deepfakes); this
> module makes the technique concrete, not a license to use it on anyone
> else's voice without asking first.

## 1. Build

```
docker compose build voice-cloning
```

## 2. Study the code directly

```
docker compose run --rm voice-cloning python core.py \
  "hello, this is a test of voice cloning" \
  /data/samples/hello.wav \
  /data/voice_cloning_output.wav
```

The second argument is the reference clip whose voice gets cloned (a few
seconds is enough — no fine-tuning happens). First run downloads the model
into `models_cache/voice-cloning`.

## 3. Run the API

```
docker compose up -d voice-cloning
curl http://localhost:8006/health
```

```
curl -X POST http://localhost:8006/clone \
  -F "text=hello, this is a test of voice cloning" \
  -F "language=en" \
  -F "file=@data/samples/hello.wav" \
  --output data/voice_cloning_result.wav
```

```
docker compose down
```

## Notes

- Compare the output against `text-to-speech`'s generic voices — this
  module should sound noticeably closer to whoever's voice was in the
  reference clip.
- Lesson 10's evaluation metric (speaker-similarity via a verification
  embedding) can be checked directly: run the cloned output back through
  `speaker-verification`'s `/verify` against the original reference speaker.
