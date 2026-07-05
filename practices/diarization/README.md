# diarization — lesson 06

Covers [06 · Speaker Diarization](../../lessons/06-diarization.html) using
pyannote.audio's pretrained pipeline. GPU-capable.

## 0. One-time setup: Hugging Face token (required, manual)

This pipeline's weights are gated — you need to accept its license yourself
before a token can download it:

1. Create a free account at [huggingface.co](https://huggingface.co) if you don't have one.
2. Visit [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the license (there's also a dependent model, `pyannote/segmentation-3.0` — accept that one too if prompted).
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (a read-only token is enough).
4. In the repo root, `cp .env.example .env` if you haven't already, then set:
   ```
   HF_TOKEN=hf_your_token_here
   ```

Without this, `core.get_pipeline()` raises a clear error telling you the
same thing.

## 1. Build

```
docker compose build diarization
```

## 2. Study the code directly

```
docker compose run --rm diarization python core.py /data/samples/hello.wav
```

Prints the device in use, then each detected speaker turn with timestamps.
First run downloads the pipeline's weights into
`models_cache/diarization` — later runs reuse that cache.

## 3. Run the API

```
docker compose up -d diarization
curl http://localhost:8002/health
```

```
curl -X POST http://localhost:8002/diarize \
  -F "file=@data/samples/hello.wav"
```

Returns:
```json
{
  "segments": [{"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00"}],
  "num_speakers": 1,
  "saved_input_file": "/data/diarization_upload_....wav"
}
```

```
docker compose down
```

## Notes

- A single-speaker clip (like the checked-in sample) will trivially report one speaker — this module is most interesting on a multi-speaker recording. Try feeding it a short multi-person clip once you have one.
- Pairing this output with `speech-to-text`'s transcript (matching timestamps) is exactly what lesson 11's capstone script does.
