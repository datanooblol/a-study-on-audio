FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# espeak-ng backs the grapheme-to-phoneme step some Coqui models use.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY requirements.txt ./
RUN uv pip install --system -r requirements.txt

COPY . .

# Coqui prompts interactively to accept a model's license the first time it
# downloads it; this skips that prompt (you're accepting on the model's own
# terms by choosing to run it here -- check the model card if unsure).
ENV COQUI_TOS_AGREED=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
