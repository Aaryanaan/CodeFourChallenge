"""Centralized configuration via Pydantic Settings.

All settings are loaded from environment variables and .env file.
Field names map directly to uppercase env var names (e.g.,
video_dir -> VIDEO_DIR, openrouter_api_key -> OPENROUTER_API_KEY).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Paths (per D-12)
    video_dir: Path = Path("data/videos")
    metadata_dir: Path = Path("data/metadata")
    index_dir: Path = Path("data/index")

    # ffmpeg
    ffmpeg_path: str = "ffmpeg"

    # PySceneDetect (per D-10)
    pyscenedetect_threshold: float = 40.0
    chunk_min_duration: float = 10.0
    chunk_max_duration: float = 60.0
    sliding_window_size: float = 30.0
    sliding_window_overlap: float = 10.0

    # API keys (per D-11)
    openrouter_api_key: str = ""
    google_api_key: str = ""

    # Model identifiers (per D-11, ARC-03)
    captioner_model: str = "google/gemini-2.5-flash"
    reranker_model: str = "anthropic/claude-sonnet-4"
    embedder_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    embedding_batch_size: int = 50

    # Extraction settings
    whisper_model: str = "large-v3"  # faster-whisper model size
    whisper_compute_type: str = "auto"  # auto selects best for platform
    ocr_confidence_threshold: float = 0.7  # minimum PaddleOCR confidence
    raised_voice_stddev_threshold: float = 2.0  # RMS std devs above mean
    ocr_frame_interval: float = 2.0  # seconds between OCR frame samples

    # Retrieval weights (Phase 4 defaults, Phase 6 overrides per-query)
    retrieval_vector_weight: float = 1.0
    retrieval_bm25_weight: float = 1.0
    retrieval_filter_weight: float = 0.5

    # Captioning (Phase 5)
    caption_cost_ceiling: float = 5.0
    caption_cache_dir: Path = Path("data/cache/captions")
    caption_cost_per_chunk: float = 0.003

    # Query Intelligence (Phase 6)
    classifier_model: str = "gemini-2.0-flash"  # Google AI SDK model ID (no google/ prefix)
    classifier_cache_dir: Path = Path("data/cache/classifier")
    reranker_cache_dir: Path = Path("data/cache/reranker")
