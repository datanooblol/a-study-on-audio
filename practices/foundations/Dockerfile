FROM python:3.11-slim

# System deps for lesson 02's format conversion (ffmpeg/sox) and for
# soundfile/librosa's file I/O (libsndfile1) — this is exactly the kind of
# install that's painful on native Windows and trivial in a container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml ./
RUN uv sync

COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
