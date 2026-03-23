"""Tests for audio extraction utility."""

import os

import pytest
import soundfile as sf

from videosearch.audio_utils import extract_audio_segment


def test_extract_audio_segment_produces_wav(test_video):
    """extract_audio_segment produces a .wav file that exists on disk."""
    wav_path = extract_audio_segment(test_video, 0.0, 5.0)
    try:
        assert os.path.exists(wav_path)
        assert wav_path.endswith(".wav")
    finally:
        os.unlink(wav_path)


def test_extract_audio_segment_mono_16khz(test_video):
    """Extracted WAV is mono (1 channel) with 16000 Hz sample rate."""
    wav_path = extract_audio_segment(test_video, 0.0, 5.0)
    try:
        info = sf.info(wav_path)
        assert info.channels == 1, f"Expected 1 channel, got {info.channels}"
        assert info.samplerate == 16000, f"Expected 16000 Hz, got {info.samplerate}"
    finally:
        os.unlink(wav_path)


def test_extract_audio_segment_invalid_path():
    """extract_audio_segment with invalid video path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_audio_segment("/nonexistent/video.mp4", 0.0, 5.0)
