"""Thin FastAPI wrapper around core.py's transcribe() -- see core.py for the actual logic."""
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core

app = FastAPI(title="speech-to-text — lesson 05")
DATA_DIR = Path("/data")


@app.on_event("startup")
def _warm_up():
    # Load the model once at startup rather than on the first request, so
    # the first real /transcribe call isn't the one paying for model load.
    core.get_model()


@app.get("/health")
def health():
    return {"status": "ok", "device": core.get_device()}


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...), model_size: str = Form(None)):
    raw = await file.read()
    tmp_path = DATA_DIR / f"speech_to_text_upload_{int(time.time())}_{file.filename}"
    tmp_path.write_bytes(raw)

    result = core.transcribe(str(tmp_path), model_size)
    result["saved_input_file"] = str(tmp_path)
    return result
