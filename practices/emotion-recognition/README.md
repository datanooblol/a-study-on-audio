# emotion-recognition — lesson 08

Covers [08 · Speech Emotion Recognition](../../lessons/08-emotion-recognition.html)
using a wav2vec2 model fine-tuned for categorical emotion classification
(`superb/wav2vec2-base-superb-er`, trained on IEMOCAP). GPU-capable, no
gating/token required.

## 1. Build

```
docker compose build emotion-recognition
```

## 2. Study the code directly

```
docker compose run --rm emotion-recognition python core.py /data/samples/hello.wav
```

Prints the predicted emotion label and the full score distribution across
classes. First run downloads the model into `models_cache/emotion-recognition`.

## 3. Run the API

```
docker compose up -d emotion-recognition
curl http://localhost:8004/health
```

```
curl -X POST http://localhost:8004/predict \
  -F "file=@data/samples/hello.wav"
```

Returns:
```json
{"emotion": "neu", "scores": {"neu": 0.81, "hap": 0.09, "ang": 0.06, "sad": 0.04}}
```

```
docker compose down
```

## Notes

- IEMOCAP (this model's training data) is **acted** emotion by actors
  reading scripted/improvised lines — see lesson 08's callout on the
  acted-vs-spontaneous domain gap before trusting this on real, unscripted
  audio. It's a good demo of the mechanics, not a production-grade
  detector out of the box.
