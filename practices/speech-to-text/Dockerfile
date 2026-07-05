FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# ffmpeg is how whisper actually decodes arbitrary input audio formats.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY requirements.txt ./
# --system: install into this image's existing Python (which already has a
# CUDA-matched torch), not a fresh uv-managed venv. See lesson 02/README for
# why re-installing torch here would be wasteful and risks a CPU-only build.
RUN uv pip install --system -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
