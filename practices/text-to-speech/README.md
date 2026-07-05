# text-to-speech — lesson 09

Covers [09 · Text-to-Speech (TTS)](../../lessons/09-text-to-speech.html) using
Coqui TTS's VITS model trained on VCTK (~100 built-in voices). GPU-capable.

## 1. Build

```
docker compose build text-to-speech
```

## 2. Study the code directly

```
docker compose run --rm text-to-speech python core.py "hello, this is a test" /data/tts_output.wav
```

Prints the device in use and a sample of available speaker IDs, then saves
the synthesized audio to `/data/tts_output.wav` (`data/tts_output.wav` on the
host — play it back there). First run downloads the model into
`models_cache/text-to-speech`.

## 3. Run the API

```
docker compose up -d text-to-speech
curl http://localhost:8005/health
curl http://localhost:8005/speakers
```

```
curl -X POST http://localhost:8005/synthesize \
  -F "text=hello, this is a test" \
  -F "speaker=p225" \
  --output data/tts_result.wav
```

```
docker compose down
```

## Notes

- Omit `speaker` to get the model's default voice — call `/speakers` first to see the full list of ~100 VCTK voice IDs.
- This is the "generic multi-speaker TTS" module; lesson 10's `voice-cloning` is the one that adapts to an arbitrary *new* voice from a short reference clip, not just picking among VCTK's built-in speakers.
