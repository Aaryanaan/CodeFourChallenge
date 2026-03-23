"""Tests for WhisperTranscriber with mocked Whisper model.

Tests verify protocol conformance, segment structure, hallucination
filtering (avg_logprob < -1.0), no_speech_prob filtering (> 0.6),
word-level timestamps, language detection, and temp file cleanup.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from videosearch.protocols import Transcriber
from videosearch.transcriber import WhisperTranscriber


def _make_segment(
    text="hello world",
    start=0.0,
    end=2.0,
    avg_logprob=-0.5,
    no_speech_prob=0.1,
    words=None,
):
    """Build a mock faster-whisper segment object."""
    if words is None:
        words = [
            SimpleNamespace(word="hello", start=0.0, end=1.0, probability=0.95),
            SimpleNamespace(word="world", start=1.0, end=2.0, probability=0.90),
        ]
    seg = SimpleNamespace(
        text=text,
        start=start,
        end=end,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        words=words,
    )
    return seg


def _make_info(language="en", language_probability=0.98):
    """Build a mock faster-whisper transcription info object."""
    return SimpleNamespace(language=language, language_probability=language_probability)


@pytest.fixture
def mock_whisper_model():
    """Fixture that patches WhisperModel to avoid real model download."""
    with patch("videosearch.transcriber.WhisperModel") as MockModel:
        instance = MagicMock()
        MockModel.return_value = instance
        yield instance


@pytest.fixture
def mock_audio_extract(tmp_path):
    """Fixture that patches extract_audio_segment to return a dummy WAV path."""
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"\x00" * 100)
    with patch("videosearch.transcriber.extract_audio_segment") as mock_fn:
        mock_fn.return_value = str(dummy_wav)
        yield mock_fn


class TestWhisperTranscriberProtocol:
    """Test 1: Protocol conformance."""

    def test_satisfies_transcriber_protocol(self, mock_whisper_model):
        transcriber = WhisperTranscriber(model_size="tiny")
        assert isinstance(transcriber, Transcriber)


class TestTranscribeReturnStructure:
    """Tests 2, 3, 6, 7: Return dict structure and segment/word fields."""

    def test_returns_dict_with_segments_key(
        self, mock_whisper_model, mock_audio_extract
    ):
        segments = [_make_segment()]
        info = _make_info()
        mock_whisper_model.transcribe.return_value = (iter(segments), info)

        transcriber = WhisperTranscriber(model_size="tiny")
        result = transcriber.transcribe("video.mp4", 0.0, 5.0)

        assert isinstance(result, dict)
        assert "segments" in result
        assert isinstance(result["segments"], list)

    def test_segment_dict_has_required_keys(
        self, mock_whisper_model, mock_audio_extract
    ):
        segments = [_make_segment()]
        info = _make_info()
        mock_whisper_model.transcribe.return_value = (iter(segments), info)

        transcriber = WhisperTranscriber(model_size="tiny")
        result = transcriber.transcribe("video.mp4", 0.0, 5.0)

        seg = result["segments"][0]
        assert "text" in seg
        assert "start" in seg
        assert "end" in seg
        assert "avg_logprob" in seg
        assert "words" in seg

    def test_returns_language_and_probability(
        self, mock_whisper_model, mock_audio_extract
    ):
        segments = [_make_segment()]
        info = _make_info(language="en", language_probability=0.98)
        mock_whisper_model.transcribe.return_value = (iter(segments), info)

        transcriber = WhisperTranscriber(model_size="tiny")
        result = transcriber.transcribe("video.mp4", 0.0, 5.0)

        assert "language" in result
        assert "language_probability" in result
        assert result["language"] == "en"
        assert result["language_probability"] == 0.98

    def test_words_list_has_required_keys(
        self, mock_whisper_model, mock_audio_extract
    ):
        segments = [_make_segment()]
        info = _make_info()
        mock_whisper_model.transcribe.return_value = (iter(segments), info)

        transcriber = WhisperTranscriber(model_size="tiny")
        result = transcriber.transcribe("video.mp4", 0.0, 5.0)

        word = result["segments"][0]["words"][0]
        assert "word" in word
        assert "start" in word
        assert "end" in word
        assert "probability" in word


class TestHallucinationFilter:
    """Test 4: Segments with avg_logprob < -1.0 are filtered out."""

    def test_hallucination_filter(self, mock_whisper_model, mock_audio_extract):
        good_seg = _make_segment(text="good segment", avg_logprob=-0.3)
        bad_seg = _make_segment(text="hallucinated", avg_logprob=-1.5)
        borderline_seg = _make_segment(text="borderline", avg_logprob=-1.0)
        info = _make_info()
        mock_whisper_model.transcribe.return_value = (
            iter([good_seg, bad_seg, borderline_seg]),
            info,
        )

        transcriber = WhisperTranscriber(model_size="tiny")
        result = transcriber.transcribe("video.mp4", 0.0, 10.0)

        texts = [s["text"] for s in result["segments"]]
        assert "good segment" in texts
        assert "hallucinated" not in texts
        # Borderline (exactly -1.0) should pass since filter is strictly < -1.0
        assert "borderline" in texts


class TestNoSpeechFilter:
    """Test 5: Segments with no_speech_prob > 0.6 are filtered out."""

    def test_no_speech_filter(self, mock_whisper_model, mock_audio_extract):
        speech_seg = _make_segment(text="real speech", no_speech_prob=0.1)
        noise_seg = _make_segment(text="noise artifact", no_speech_prob=0.8)
        info = _make_info()
        mock_whisper_model.transcribe.return_value = (
            iter([speech_seg, noise_seg]),
            info,
        )

        transcriber = WhisperTranscriber(model_size="tiny")
        result = transcriber.transcribe("video.mp4", 0.0, 10.0)

        texts = [s["text"] for s in result["segments"]]
        assert "real speech" in texts
        assert "noise artifact" not in texts


class TestTempFileCleanup:
    """Test 8: Temp audio file is cleaned up even when processing fails."""

    def test_cleanup_on_success(self, mock_whisper_model, tmp_path):
        dummy_wav = tmp_path / "cleanup_test.wav"
        dummy_wav.write_bytes(b"\x00" * 100)

        with patch("videosearch.transcriber.extract_audio_segment") as mock_extract:
            mock_extract.return_value = str(dummy_wav)
            info = _make_info()
            mock_whisper_model.transcribe.return_value = (
                iter([_make_segment()]),
                info,
            )

            transcriber = WhisperTranscriber(model_size="tiny")
            transcriber.transcribe("video.mp4", 0.0, 5.0)

        assert not dummy_wav.exists(), "Temp audio file should be cleaned up"

    def test_cleanup_on_failure(self, mock_whisper_model, tmp_path):
        dummy_wav = tmp_path / "cleanup_fail_test.wav"
        dummy_wav.write_bytes(b"\x00" * 100)

        with patch("videosearch.transcriber.extract_audio_segment") as mock_extract:
            mock_extract.return_value = str(dummy_wav)
            mock_whisper_model.transcribe.side_effect = RuntimeError("model error")

            transcriber = WhisperTranscriber(model_size="tiny")
            with pytest.raises(RuntimeError, match="model error"):
                transcriber.transcribe("video.mp4", 0.0, 5.0)

        assert not dummy_wav.exists(), "Temp audio file should be cleaned up on error"
