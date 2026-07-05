"""Thin FastAPI wrapper around core.py's enroll/verify/identify -- see core.py for the actual logic."""
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core

app = FastAPI(title="speaker-verification — lesson 07")
DATA_DIR = Path("/data")


def _save_upload(file_bytes: bytes, filename: str) -> Path:
    path = DATA_DIR / f"speaker_verification_upload_{int(time.time())}_{filename}"
    path.write_bytes(file_bytes)
    return path


@app.get("/health")
def health():
    return {"status": "ok", "device": core.get_device()}


@app.post("/enroll")
async def enroll_endpoint(speaker_name: str = Form(...), file: UploadFile = File(...)):
    path = _save_upload(await file.read(), file.filename)
    return core.enroll(speaker_name, str(path))


@app.post("/verify")
async def verify_endpoint(speaker_name: str = Form(...), file: UploadFile = File(...)):
    path = _save_upload(await file.read(), file.filename)
    return core.verify(speaker_name, str(path))


@app.post("/identify")
async def identify_endpoint(file: UploadFile = File(...)):
    path = _save_upload(await file.read(), file.filename)
    return core.identify(str(path))
