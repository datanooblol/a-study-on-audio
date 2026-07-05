# speech-to-text — lesson 05

Covers [05 · Speech-to-Text (ASR)](../../lessons/05-speech-to-text.html) using
OpenAI's Whisper. GPU-capable (falls back to CPU automatically if no GPU is
available, just much slower).

## 1. Build

```
docker compose build speech-to-text
```

## 2. Study the code directly

```
docker compose run --rm speech-to-text python core.py /data/samples/hello.wav
```
(Note: plain `python`, not `uv run` — this module installs its extra
dependencies straight into the base image's existing Python via
`uv pip install --system`, rather than managing a separate uv venv, so it can
reuse the base image's CUDA-matched torch. See the Dockerfile comments.)

Prints which device it's running on (`cuda` or `cpu`), then the transcript
with per-segment timestamps. First run downloads the model
(`WHISPER_MODEL`, default `base`) into `models_cache/speech-to-text` — later
runs reuse that cache.

## 3. Run the API

```
docker compose up -d speech-to-text
curl http://localhost:8001/health
```

```
curl -X POST http://localhost:8001/transcribe \
  -F "file=@data/samples/hello.wav"
```

Returns:
```json
{
  "text": "...",
  "language": "en",
  "segments": [{"start": 0.0, "end": 1.8, "text": "..."}],
  "saved_input_file": "/data/speech_to_text_upload_....wav"
}
```

```
docker compose down
```

## Notes

- Change model size via `.env`'s `WHISPER_MODEL` (`tiny`/`base`/`small`/`medium`/`large`) — bigger is slower but more accurate.
- To confirm the GPU is actually being used: `/health` reports the device, or watch `nvidia-smi` on the host while a request is in flight.
