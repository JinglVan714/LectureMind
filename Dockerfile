FROM python:3.11-slim

# System deps:
#   ffmpeg  → keyframe / audio extraction
#   git, build-essential → some pip wheels (opencv, scenedetect)
#   curl     → healthcheck convenience
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY app /app/app

RUN pip install --upgrade pip \
    && pip install -e ".[whisper]"

# Default DB and reports under /app/data; mount a volume to persist
ENV DATA_DIR=/app/data \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["python", "-m", "app.main"]
