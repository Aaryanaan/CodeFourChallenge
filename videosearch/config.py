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
