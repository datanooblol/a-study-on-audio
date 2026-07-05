"""Thin FastAPI wrapper around core.py's clone() -- see core.py for the actual logic."""
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core

app = FastAPI(title="voice-cloning — lesson 10")
DATA_DIR = Path("/data")


@app.get("/health")
def health():
    return {"status": "ok", "device": core.get_device()}


@app.post("/clone")
async def clone_endpoint(
    text: str = Form(...),
    language: str = Form("en"),
    file: UploadFile = File(...),
):
    ref_path = DATA_DIR / f"voice_cloning_reference_{int(time.time())}_{file.filename}"
    ref_path.write_bytes(await file.read())

    out_path = DATA_DIR / f"voice_cloning_output_{int(time.time())}.wav"
    core.clone(text, str(ref_path), str(out_path), language)
    return FileResponse(out_path, media_type="audio/wav", filename=out_path.name)
