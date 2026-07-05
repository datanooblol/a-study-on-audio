"""Thin FastAPI wrapper around core.py's diarize() -- see core.py for the actual logic."""
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core

app = FastAPI(title="diarization — lesson 06")
DATA_DIR = Path("/data")


@app.get("/health")
def health():
    return {"status": "ok", "device": core.get_device()}


@app.post("/diarize")
async def diarize_endpoint(file: UploadFile = File(...)):
    raw = await file.read()
    tmp_path = DATA_DIR / f"diarization_upload_{int(time.time())}_{file.filename}"
    tmp_path.write_bytes(raw)

    result = core.diarize(str(tmp_path))
    result["saved_input_file"] = str(tmp_path)
    return result
