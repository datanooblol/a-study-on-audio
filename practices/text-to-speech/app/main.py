"""Thin FastAPI wrapper around core.py's synthesize() -- see core.py for the actual logic."""
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core

app = FastAPI(title="text-to-speech — lesson 09")
DATA_DIR = Path("/data")


@app.get("/health")
def health():
    return {"status": "ok", "device": core.get_device()}


@app.get("/speakers")
def speakers():
    return {"speakers": core.list_speakers()}


@app.post("/synthesize")
def synthesize_endpoint(text: str = Form(...), speaker: str = Form(None)):
    out_path = DATA_DIR / f"text_to_speech_{int(time.time())}.wav"
    core.synthesize(text, str(out_path), speaker)
    return FileResponse(out_path, media_type="audio/wav", filename=out_path.name)
