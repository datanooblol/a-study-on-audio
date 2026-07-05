# speaker-verification — lesson 07

Covers [07 · Speaker Verification & Identification](../../lessons/07-speaker-verification.html)
using speechbrain's ECAPA-TDNN embedding model. GPU-capable.

## 1. Build

```
docker compose build speaker-verification
```

## 2. Study the code directly

```
docker compose run --rm speaker-verification python core.py enroll alice /data/samples/hello.wav
docker compose run --rm speaker-verification python core.py verify alice /data/samples/hello.wav
docker compose run --rm speaker-verification python core.py identify /data/samples/hello.wav
```

Enrolled voiceprints are stored in `data/voiceprints.json` on the host (plain
JSON — fine for a sandbox, not how you'd store biometric data for real).
First run downloads the embedding model into `models_cache/speaker-verification`.

## 3. Run the API

```
docker compose up -d speaker-verification
curl http://localhost:8003/health
```

```
curl -X POST http://localhost:8003/enroll \
  -F "speaker_name=alice" -F "file=@data/samples/hello.wav"

curl -X POST http://localhost:8003/verify \
  -F "speaker_name=alice" -F "file=@data/samples/hello.wav"

curl -X POST http://localhost:8003/identify \
  -F "file=@data/samples/hello.wav"
```

```
docker compose down
```

## Notes

- `DEFAULT_THRESHOLD = 0.25` in `core.py` is a rough starting point for the
  accept/reject cosine-similarity cutoff — lesson 07's EER is the principled
  way to tune this against your own enrolled voices; this sandbox doesn't
  compute it for you.
- Enrolling and testing with the *same* short clip (as in the example above)
  will trivially "verify" — for a meaningful test, enroll with one recording
  and verify with a different recording of the same person, plus a
  recording of someone else, to see the score actually discriminate.
