"""Unit tests for GeminiCaptioner with mocked API."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from videosearch.captioner import GeminiCaptioner
from videosearch.config import Settings
from videosearch.protocols import Captioner


@pytest.fixture
def mock_settings(tmp_path):
    """Settings with test values pointing to tmp_path for cache."""
    settings = MagicMock(spec=Settings)
    settings.google_api_key = "test-api-key"
    settings.captioner_model = "google/gemini-2.5-flash"
    settings.caption_cache_dir = tmp_path / "cache" / "captions"
    settings.ffmpeg_path = "ffmpeg"
    return settings


@pytest.fixture
def mock_genai_client():
    """Mock google.genai.Client."""
    with patch("videosearch.captioner.genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        yield mock_genai, mock_client


def _make_captioner(mock_settings, mock_genai_client):
    """Helper to create a GeminiCaptioner with mocked client."""
    mock_genai, mock_client = mock_genai_client
    captioner = GeminiCaptioner(mock_settings)
    return captioner, mock_client


def test_captioner_satisfies_protocol(mock_settings, mock_genai_client):
    """GeminiCaptioner satisfies the Captioner protocol."""
    captioner, _ = _make_captioner(mock_settings, mock_genai_client)
    assert isinstance(captioner, Captioner)


def test_caption_returns_dict_with_caption_key(mock_settings, mock_genai_client, tmp_path):
    """caption() returns dict with 'caption' key containing labeled text."""
    captioner, mock_client = _make_captioner(mock_settings, mock_genai_client)

    # Mock uploaded file
    mock_uploaded = MagicMock()
    mock_uploaded.name = "files/test-file-id"
    mock_client.files.upload.return_value = mock_uploaded

    # Mock generate_content response
    mock_response = MagicMock()
    mock_response.text = "Clothing: navy uniform\nActions: walking\nObjects: vehicle\nText: none\nLighting: daylight"
    mock_client.models.generate_content.return_value = mock_response

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close") as mock_close, \
         patch("os.unlink") as mock_unlink:
        mock_mkstemp.return_value = (5, str(tmp_path / "tmp_clip.mp4"))
        mock_run.return_value = MagicMock(returncode=0)

        result = captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    assert "caption" in result
    assert "Clothing:" in result["caption"]


def test_caption_calls_ffmpeg_with_correct_flags(mock_settings, mock_genai_client, tmp_path):
    """caption() calls ffmpeg with -ss, -t, -c copy, -an flags."""
    captioner, mock_client = _make_captioner(mock_settings, mock_genai_client)

    mock_uploaded = MagicMock()
    mock_uploaded.name = "files/test-file-id"
    mock_client.files.upload.return_value = mock_uploaded

    mock_response = MagicMock()
    mock_response.text = "Clothing: test\nActions: test\nObjects: test\nText: none\nLighting: day"
    mock_client.models.generate_content.return_value = mock_response

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink"):
        tmp_clip = str(tmp_path / "tmp_clip.mp4")
        mock_mkstemp.return_value = (5, tmp_clip)
        mock_run.return_value = MagicMock(returncode=0)

        captioner.caption(str(tmp_path / "video.mp4"), 10.0, 40.0)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "-ss" in call_args
        assert "-t" in call_args
        assert "-c" in call_args
        assert "copy" in call_args
        assert "-an" in call_args


def test_caption_calls_upload_and_generate_content(mock_settings, mock_genai_client, tmp_path):
    """caption() uploads file via client.files.upload and calls generate_content."""
    captioner, mock_client = _make_captioner(mock_settings, mock_genai_client)

    mock_uploaded = MagicMock()
    mock_uploaded.name = "files/test-file-id"
    mock_client.files.upload.return_value = mock_uploaded

    mock_response = MagicMock()
    mock_response.text = "Clothing: uniform\nActions: standing\nObjects: car\nText: none\nLighting: night"
    mock_client.models.generate_content.return_value = mock_response

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink"):
        mock_mkstemp.return_value = (5, str(tmp_path / "tmp_clip.mp4"))
        mock_run.return_value = MagicMock(returncode=0)

        captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    mock_client.files.upload.assert_called_once()
    mock_client.models.generate_content.assert_called_once()
    # Verify uploaded file is deleted
    mock_client.files.delete.assert_called_once_with(name=mock_uploaded.name)


def test_caption_deletes_uploaded_file_on_generate_error(mock_settings, mock_genai_client, tmp_path):
    """caption() deletes uploaded file even when generate_content raises an exception."""
    captioner, mock_client = _make_captioner(mock_settings, mock_genai_client)

    mock_uploaded = MagicMock()
    mock_uploaded.name = "files/failing-file-id"
    mock_client.files.upload.return_value = mock_uploaded
    mock_client.models.generate_content.side_effect = RuntimeError("API error")

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink"):
        mock_mkstemp.return_value = (5, str(tmp_path / "tmp_clip.mp4"))
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(RuntimeError, match="API error"):
            captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    # File must be deleted even on error
    mock_client.files.delete.assert_called_once_with(name=mock_uploaded.name)


def test_caption_deletes_temp_clip_on_upload_error(mock_settings, mock_genai_client, tmp_path):
    """caption() deletes temp clip even when upload raises an exception."""
    captioner, mock_client = _make_captioner(mock_settings, mock_genai_client)

    mock_client.files.upload.side_effect = RuntimeError("Upload failed")

    tmp_clip = str(tmp_path / "tmp_clip.mp4")

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink") as mock_unlink:
        mock_mkstemp.return_value = (5, tmp_clip)
        mock_run.return_value = MagicMock(returncode=0)

        with pytest.raises(RuntimeError, match="Upload failed"):
            captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    # Temp clip must be deleted even on upload error
    mock_unlink.assert_called_with(tmp_clip)


def test_caption_returns_cached_value_without_api_call(mock_settings, mock_genai_client, tmp_path):
    """When cache file exists, caption() returns cached value WITHOUT calling API."""
    captioner, mock_client = _make_captioner(mock_settings, mock_genai_client)

    # Set cache dir and create a cache file for video_id="video", chunk_index=0
    cache_dir = tmp_path / "cache" / "captions" / "video"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "0.json"
    cache_data = {
        "caption": "Clothing: cached uniform\nActions: cached action",
        "model": "gemini-2.5-flash",
        "cached_at": "2024-01-01T00:00:00+00:00",
    }
    cache_file.write_text(json.dumps(cache_data))

    result = captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    # API should not be called
    mock_client.files.upload.assert_not_called()
    mock_client.models.generate_content.assert_not_called()

    assert result["caption"] == cache_data["caption"]
    assert result.get("cached") is True


def test_caption_writes_cache_on_miss(mock_settings, mock_genai_client, tmp_path):
    """On cache miss, caption() writes cache file with caption, model, cached_at keys."""
    captioner, mock_client = _make_captioner(mock_settings, mock_genai_client)

    mock_uploaded = MagicMock()
    mock_uploaded.name = "files/new-file-id"
    mock_client.files.upload.return_value = mock_uploaded

    caption_text = "Clothing: blue jacket\nActions: running\nObjects: bicycle\nText: none\nLighting: overcast"
    mock_response = MagicMock()
    mock_response.text = caption_text
    mock_client.models.generate_content.return_value = mock_response

    with patch("videosearch.captioner.subprocess.run") as mock_run, \
         patch("videosearch.captioner.tempfile.mkstemp") as mock_mkstemp, \
         patch("os.close"), \
         patch("os.unlink"):
        mock_mkstemp.return_value = (5, str(tmp_path / "tmp_clip.mp4"))
        mock_run.return_value = MagicMock(returncode=0)

        result = captioner.caption(str(tmp_path / "video.mp4"), 0.0, 30.0)

    # Verify cache file was written
    # video_id from "video.mp4" -> stem "video" -> replace "_720p" -> "video"
    # chunk_index = int(0.0) = 0
    cache_path = tmp_path / "cache" / "captions" / "video" / "0.json"
    assert cache_path.exists()

    with cache_path.open() as f:
        cached = json.load(f)

    assert cached["caption"] == caption_text
    assert "model" in cached
    assert "cached_at" in cached
    assert result["cached"] is False
