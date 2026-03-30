FROM python:3.11-slim

# System deps:
# - ffmpeg: video compression, chunking, audio extraction
# - libgl1, libglib2.0-0: required by OpenCV (used by PaddleOCR and PySceneDetect)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY videosearch/ ./videosearch/
COPY data/metadata/ ./data/metadata/
COPY data/index/ ./data/index/

RUN pip install --no-cache-dir -e ".[dev]"

# faster-whisper large-v3 (~3 GB) downloads on first ingest.
# Pre-download here to bake it into the image (optional — adds ~3 GB to image size).
# Uncomment the line below if you want offline ingestion support:
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', compute_type='auto')"

CMD ["videosearch", "--help"]
