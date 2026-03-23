"""Tests for Settings configuration."""

from pathlib import Path

from videosearch.config import Settings


def test_settings_defaults():
    """Settings instantiates with correct defaults when no env vars are set."""
    settings = Settings(_env_file=None)

    # Paths
    assert settings.video_dir == Path("data/videos")
    assert settings.metadata_dir == Path("data/metadata")
    assert settings.index_dir == Path("data/index")

    # ffmpeg
    assert settings.ffmpeg_path == "ffmpeg"

    # PySceneDetect
    assert settings.pyscenedetect_threshold == 40.0
    assert settings.chunk_min_duration == 10.0
    assert settings.chunk_max_duration == 60.0
    assert settings.sliding_window_size == 30.0
    assert settings.sliding_window_overlap == 10.0

    # API keys default to empty
    assert settings.openrouter_api_key == ""
    assert settings.google_api_key == ""

    # Model identifiers
    assert settings.captioner_model == "google/gemini-2.5-flash"
    assert settings.reranker_model == "anthropic/claude-sonnet-4"
    assert settings.embedder_model == "text-embedding-004"


def test_settings_api_key_from_env(monkeypatch):
    """OPENROUTER_API_KEY env var is loaded into settings."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key_123")
    settings = Settings(_env_file=None)
    assert settings.openrouter_api_key == "test_key_123"


def test_settings_google_api_key_from_env(monkeypatch):
    """GOOGLE_API_KEY env var is loaded into settings."""
    monkeypatch.setenv("GOOGLE_API_KEY", "goog_key_456")
    settings = Settings(_env_file=None)
    assert settings.google_api_key == "goog_key_456"


def test_model_config_from_env(monkeypatch):
    """Model identifier env vars override defaults."""
    monkeypatch.setenv("CAPTIONER_MODEL", "other/model")
    monkeypatch.setenv("RERANKER_MODEL", "other/reranker")
    monkeypatch.setenv("EMBEDDER_MODEL", "other-embed")
    settings = Settings(_env_file=None)
    assert settings.captioner_model == "other/model"
    assert settings.reranker_model == "other/reranker"
    assert settings.embedder_model == "other-embed"


def test_path_config_from_env(monkeypatch):
    """Path env vars override defaults and are converted to Path objects."""
    monkeypatch.setenv("VIDEO_DIR", "/tmp/vids")
    monkeypatch.setenv("METADATA_DIR", "/tmp/meta")
    monkeypatch.setenv("INDEX_DIR", "/tmp/idx")
    settings = Settings(_env_file=None)
    assert settings.video_dir == Path("/tmp/vids")
    assert settings.metadata_dir == Path("/tmp/meta")
    assert settings.index_dir == Path("/tmp/idx")


def test_pyscenedetect_config_from_env(monkeypatch):
    """PySceneDetect numeric config loaded from env vars as floats."""
    monkeypatch.setenv("PYSCENEDETECT_THRESHOLD", "50.0")
    monkeypatch.setenv("CHUNK_MIN_DURATION", "15.0")
    monkeypatch.setenv("CHUNK_MAX_DURATION", "90.0")
    monkeypatch.setenv("SLIDING_WINDOW_SIZE", "45.0")
    monkeypatch.setenv("SLIDING_WINDOW_OVERLAP", "15.0")
    settings = Settings(_env_file=None)
    assert settings.pyscenedetect_threshold == 50.0
    assert settings.chunk_min_duration == 15.0
    assert settings.chunk_max_duration == 90.0
    assert settings.sliding_window_size == 45.0
    assert settings.sliding_window_overlap == 15.0


# --- Extraction settings tests ---


def test_settings_whisper_model_default():
    """Settings has whisper_model field defaulting to 'large-v3'."""
    settings = Settings(_env_file=None)
    assert settings.whisper_model == "large-v3"


def test_settings_ocr_confidence_threshold_default():
    """Settings has ocr_confidence_threshold field defaulting to 0.7."""
    settings = Settings(_env_file=None)
    assert settings.ocr_confidence_threshold == 0.7


def test_settings_raised_voice_stddev_threshold_default():
    """Settings has raised_voice_stddev_threshold field defaulting to 2.0."""
    settings = Settings(_env_file=None)
    assert settings.raised_voice_stddev_threshold == 2.0


def test_settings_whisper_compute_type_default():
    """Settings has whisper_compute_type field defaulting to 'auto'."""
    settings = Settings(_env_file=None)
    assert settings.whisper_compute_type == "auto"


def test_settings_ocr_frame_interval_default():
    """Settings has ocr_frame_interval field defaulting to 2.0."""
    settings = Settings(_env_file=None)
    assert settings.ocr_frame_interval == 2.0
