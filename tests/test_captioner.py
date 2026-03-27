"""Unit tests for GeminiCaptioner with mocked OpenRouter API."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from videosearch.captioner import GeminiCaptioner, QuotaExhaustedError
from videosearch.config import Settings
from videosearch.protocols import Captioner


@pytest.fixture
def mock_settings(tmp_path):
    """Settings with test values pointing to tmp_path for cache."""
    settings = MagicMock(spec=Settings)
    settings.openrouter_api_key = "test-openrouter-key"
    settings.captioner_model = "google/gemini-2.5-flash"
    settings.caption_cache_dir = tmp_path / "cache" / "captions"
    settings.ffmpeg_path = "ffmpeg"
    return settings


def _make_captioner(mock_settings):
    """Helper to create a GeminiCaptioner with mocked settings."""
    return GeminiCaptioner(mock_settings)


def _mock_openrouter_response(caption_text: str) -> MagicMock:
    """Create a mock httpx response for OpenRouter."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": caption_text}}]
    }
    return resp


def test_captioner_satisfies_protocol(mock_settings):
    """GeminiCaptioner satisfies the Captioner protocol."""
    captioner = _make_captioner(mock_settings)
    assert isinstance(captioner, Captioner)


def test_caption_returns_dict_with_caption_key(mock_settings, tmp_path):
    """caption() returns dict with 'caption' key containing labeled text."""
    captioner = _make_captioner(mock_settings)
    caption_text = "Clothing: navy uniform\nActions: walking\nObjects: vehicle\nText: none\nLighting: daylight"

    clip_path = tmp_path / "tmp_clip.mp4"
    clip_path.write_bytes(b"fake video data")

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink"), \
         patch("videosearch.captioner.httpx.post", return_value=_mock_openrouter_response(caption_text)):
        mock_mkstemp.return_value = (5, str(clip_path))
        mock_run.return_value = MagicMock(returncode=0)

        result = captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    assert "caption" in result
    assert "Clothing:" in result["caption"]


def test_caption_calls_ffmpeg_with_correct_flags(mock_settings, tmp_path):
    """caption() calls ffmpeg with -ss after -i for frame-accurate seeking, -an, and -crf."""
    captioner = _make_captioner(mock_settings)
    caption_text = "Clothing: test\nActions: test\nObjects: test\nText: none\nLighting: day"

    clip_path = tmp_path / "tmp_clip.mp4"
    clip_path.write_bytes(b"fake video data")

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink"), \
         patch("videosearch.captioner.httpx.post", return_value=_mock_openrouter_response(caption_text)):
        mock_mkstemp.return_value = (5, str(clip_path))
        mock_run.return_value = MagicMock(returncode=0)

        captioner.caption(str(tmp_path / "video.mp4"), 10.0, 40.0)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        # -i must come BEFORE -ss for frame-accurate seeking (not keyframe-aligned)
        i_idx = call_args.index("-i")
        ss_idx = call_args.index("-ss")
        assert i_idx < ss_idx, "-ss must come after -i for frame-accurate seek"
        assert "-t" in call_args
        assert "-an" in call_args
        assert "-crf" in call_args


def test_caption_calls_openrouter(mock_settings, tmp_path):
    """caption() sends request to OpenRouter API."""
    captioner = _make_captioner(mock_settings)
    caption_text = "Clothing: uniform\nActions: standing\nObjects: car\nText: none\nLighting: night"

    clip_path = tmp_path / "tmp_clip.mp4"
    clip_path.write_bytes(b"fake video data")

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink"), \
         patch("videosearch.captioner.httpx.post", return_value=_mock_openrouter_response(caption_text)) as mock_post:
        mock_mkstemp.return_value = (5, str(clip_path))
        mock_run.return_value = MagicMock(returncode=0)

        captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "openrouter.ai" in call_kwargs[0][0]


def test_caption_deletes_temp_clip_on_api_error(mock_settings, tmp_path):
    """caption() deletes temp clip even when API raises an exception."""
    captioner = _make_captioner(mock_settings)

    clip_path = tmp_path / "tmp_clip.mp4"
    clip_path.write_bytes(b"fake video data")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = RuntimeError("API error")

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink") as mock_unlink, \
         patch("videosearch.captioner.httpx.post", return_value=mock_response):
        mock_mkstemp.return_value = (5, str(clip_path))
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(RuntimeError, match="API error"):
            captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    mock_unlink.assert_called_with(str(clip_path))


def test_caption_returns_cached_value_without_api_call(mock_settings, tmp_path):
    """When cache file exists, caption() returns cached value WITHOUT calling API."""
    captioner = _make_captioner(mock_settings)

    # Set cache dir and create a cache file for video_id="video", chunk_index=0
    cache_dir = tmp_path / "cache" / "captions" / "video"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "0.json"
    cache_data = {
        "caption": "Clothing: cached uniform\nActions: cached action",
        "model": "google/gemini-2.5-flash",
        "cached_at": "2024-01-01T00:00:00+00:00",
    }
    cache_file.write_text(json.dumps(cache_data))

    with patch("videosearch.captioner.httpx.post") as mock_post:
        result = captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    # API should not be called
    mock_post.assert_not_called()

    assert result["caption"] == cache_data["caption"]
    assert result.get("cached") is True


def test_caption_chunk_index_controls_cache_key(mock_settings, tmp_path):
    """When chunk_index is passed explicitly, it -- not int(start) -- determines the cache key."""
    captioner = _make_captioner(mock_settings)

    # Pre-populate cache for chunk_index=7 (start=0.0, so int(start) would be 0)
    cache_dir = tmp_path / "cache" / "captions" / "video"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "7.json"
    cache_file.write_text('{"caption": "cached via chunk_index", "model": "x", "cached_at": "2024-01-01T00:00:00+00:00"}')

    result = captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0, chunk_index=7)

    # Should hit cache (chunk_index=7), not call API
    assert result["cached"] is True
    assert result["caption"] == "cached via chunk_index"

    # Verify chunk_index=0 cache does NOT exist (proving key was 7, not int(0.0)=0)
    assert not (cache_dir / "0.json").exists()


def test_caption_writes_cache_on_miss(mock_settings, tmp_path):
    """On cache miss, caption() writes cache file with caption, model, cached_at keys."""
    captioner = _make_captioner(mock_settings)
    caption_text = "Clothing: blue jacket\nActions: running\nObjects: bicycle\nText: none\nLighting: overcast"

    clip_path = tmp_path / "tmp_clip.mp4"
    clip_path.write_bytes(b"fake video data")

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink"), \
         patch("videosearch.captioner.httpx.post", return_value=_mock_openrouter_response(caption_text)):
        mock_mkstemp.return_value = (5, str(clip_path))
        mock_run.return_value = MagicMock(returncode=0)

        result = captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    # Verify cache file was written
    cache_path = tmp_path / "cache" / "captions" / "video" / "0.json"
    assert cache_path.exists()

    with cache_path.open() as f:
        cached = json.load(f)

    assert cached["caption"] == caption_text
    assert "model" in cached
    assert "cached_at" in cached
    assert result["cached"] is False


def test_caption_rejects_null_cache(mock_settings, tmp_path):
    """Null/empty cached captions are treated as cache misses and purged."""
    captioner = _make_captioner(mock_settings)

    cache_dir = tmp_path / "cache" / "captions" / "video"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "0.json"
    cache_file.write_text('{"caption": null, "model": "x", "cached_at": "2024-01-01T00:00:00+00:00"}')

    caption_text = "Clothing: uniform\nActions: standing\nObjects: car\nText: none\nLighting: day"
    clip_path = tmp_path / "tmp_clip.mp4"
    clip_path.write_bytes(b"fake video data")

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink"), \
         patch("videosearch.captioner.httpx.post", return_value=_mock_openrouter_response(caption_text)):
        mock_mkstemp.return_value = (5, str(clip_path))
        mock_run.return_value = MagicMock(returncode=0)

        result = captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    # Should have called API (cache miss due to null)
    # Cache file should now contain valid caption
    assert cache_file.exists()
    new_data = json.loads(cache_file.read_text())
    assert new_data["caption"] is not None
    assert "Clothing:" in new_data["caption"]


def test_quota_exhausted_raises_without_api_key(tmp_path):
    """QuotaExhaustedError raised when no OpenRouter key configured."""
    settings = MagicMock(spec=Settings)
    settings.openrouter_api_key = ""
    settings.captioner_model = "google/gemini-2.5-flash"
    settings.caption_cache_dir = tmp_path / "cache" / "captions"
    settings.ffmpeg_path = "ffmpeg"

    captioner = GeminiCaptioner(settings)

    with pytest.raises(QuotaExhaustedError):
        captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)


def test_openrouter_error_body_raises(mock_settings, tmp_path):
    """OpenRouter error in response body raises RuntimeError."""
    captioner = _make_captioner(mock_settings)

    clip_path = tmp_path / "tmp_clip.mp4"
    clip_path.write_bytes(b"fake video data")

    error_response = MagicMock()
    error_response.status_code = 200
    error_response.raise_for_status.return_value = None
    error_response.json.return_value = {"error": {"message": "model not found"}}

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink"), \
         patch("videosearch.captioner.httpx.post", return_value=error_response):
        mock_mkstemp.return_value = (5, str(clip_path))
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(RuntimeError, match="model not found"):
            captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)
