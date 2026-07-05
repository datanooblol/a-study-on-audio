"""Thin FastAPI wrapper around core.py's predict() -- see core.py for the actual logic."""
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core

app = FastAPI(title="emotion-recognition — lesson 08")
DATA_DIR = Path("/data")


@app.get("/health")
def health():
    return {"status": "ok", "device": core.get_device()}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    raw = await file.read()
    tmp_path = DATA_DIR / f"emotion_upload_{int(time.time())}_{file.filename}"
    tmp_path.write_bytes(raw)
    return core.predict(str(tmp_path))
