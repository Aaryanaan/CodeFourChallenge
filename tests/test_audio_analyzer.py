"""Tests for LibrosaAudioAnalyzer with synthetic audio data.

Tests verify protocol conformance, AudioFeatures dict structure,
RMS/pitch/ZCR value properties, raised-voice detection via per-video
relative thresholds, and temp file cleanup.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from videosearch.audio_analyzer import LibrosaAudioAnalyzer
from videosearch.protocols import AudioAnalyzer


SAMPLE_RATE = 16000


def _write_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Write a numpy array as a mono WAV file."""
    sf.write(path, audio, sr, subtype="PCM_16")


@pytest.fixture
def analyzer():
    return LibrosaAudioAnalyzer()


@pytest.fixture
def quiet_wav(tmp_path):
    """Very low amplitude audio (near-silence)."""
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.001  # Very quiet sine
    path = str(tmp_path / "quiet.wav")
    _write_wav(path, audio)
    return path


@pytest.fixture
def loud_wav(tmp_path):
    """High amplitude audio (loud sine wave at 440Hz)."""
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.9  # Loud sine
    path = str(tmp_path / "loud.wav")
    _write_wav(path, audio)
    return path


@pytest.fixture
def silence_wav(tmp_path):
    """Pure digital silence (all zeros)."""
    audio = np.zeros(SAMPLE_RATE * 2)  # 2 seconds
    path = str(tmp_path / "silence.wav")
    _write_wav(path, audio)
    return path


@pytest.fixture
def noise_wav(tmp_path):
    """Pure noise with no voiced content."""
    rng = np.random.default_rng(42)
    audio = rng.uniform(-0.3, 0.3, SAMPLE_RATE * 2).astype(np.float32)
    path = str(tmp_path / "noise.wav")
    _write_wav(path, audio)
    return path


def _mock_extract(wav_path):
    """Create a mock for extract_audio_segment that returns a given WAV path."""
    def _fn(video_path, start, end, ffmpeg_path="ffmpeg"):
        # Copy the file to a temp location so cleanup doesn't affect fixture
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        import shutil
        shutil.copy2(wav_path, tmp_path)
        return tmp_path
    return _fn


class TestLibrosaAudioAnalyzerProtocol:
    """Test 1: Protocol conformance."""

    def test_satisfies_audio_analyzer_protocol(self):
        analyzer = LibrosaAudioAnalyzer()
        assert isinstance(analyzer, AudioAnalyzer)


class TestAnalyzeReturnStructure:
    """Test 2: Return dict has all AudioFeatures fields."""

    def test_returns_dict_with_all_audio_features_keys(self, analyzer, loud_wav):
        with patch(
            "videosearch.audio_analyzer.extract_audio_segment",
            side_effect=_mock_extract(loud_wav),
        ):
            result = analyzer.analyze("video.mp4", 0.0, 2.0)

        expected_keys = {
            "rms_mean", "rms_max", "rms_stddev",
            "pitch_mean", "pitch_max", "pitch_stddev",
            "zcr_mean", "zcr_max", "zcr_stddev",
            "has_raised_voice",
        }
        assert set(result.keys()) == expected_keys


class TestRMSValues:
    """Test 3: RMS values are non-negative floats."""

    def test_rms_non_negative(self, analyzer, loud_wav):
        with patch(
            "videosearch.audio_analyzer.extract_audio_segment",
            side_effect=_mock_extract(loud_wav),
        ):
            result = analyzer.analyze("video.mp4", 0.0, 2.0)

        assert isinstance(result["rms_mean"], float)
        assert isinstance(result["rms_max"], float)
        assert isinstance(result["rms_stddev"], float)
        assert result["rms_mean"] >= 0
        assert result["rms_max"] >= 0
        assert result["rms_stddev"] >= 0


class TestPitchValues:
    """Tests 4, 5: Pitch behavior for unvoiced and voiced content."""

    def test_pitch_none_for_silence(self, analyzer, silence_wav):
        """Test 4: Pitch is None when audio has no voiced content."""
        with patch(
            "videosearch.audio_analyzer.extract_audio_segment",
            side_effect=_mock_extract(silence_wav),
        ):
            result = analyzer.analyze("video.mp4", 0.0, 2.0)

        assert result["pitch_mean"] is None
        assert result["pitch_max"] is None
        assert result["pitch_stddev"] is None

    def test_pitch_present_for_sine_wave(self, analyzer, loud_wav):
        """Test 5: Pitch values are non-None for voiced content (sine wave)."""
        with patch(
            "videosearch.audio_analyzer.extract_audio_segment",
            side_effect=_mock_extract(loud_wav),
        ):
            result = analyzer.analyze("video.mp4", 0.0, 2.0)

        assert result["pitch_mean"] is not None
        assert result["pitch_max"] is not None
        assert result["pitch_stddev"] is not None
        assert isinstance(result["pitch_mean"], float)
        assert result["pitch_mean"] > 0


class TestZCRValues:
    """Test 6: ZCR values are non-negative floats."""

    def test_zcr_non_negative(self, analyzer, loud_wav):
        with patch(
            "videosearch.audio_analyzer.extract_audio_segment",
            side_effect=_mock_extract(loud_wav),
        ):
            result = analyzer.analyze("video.mp4", 0.0, 2.0)

        assert isinstance(result["zcr_mean"], float)
        assert isinstance(result["zcr_max"], float)
        assert isinstance(result["zcr_stddev"], float)
        assert result["zcr_mean"] >= 0
        assert result["zcr_max"] >= 0
        assert result["zcr_stddev"] >= 0


class TestRaisedVoiceDetection:
    """Tests 7, 9: Raised voice detection via per-video relative thresholds."""

    def test_has_raised_voice_false_for_quiet_audio(self, analyzer, quiet_wav):
        """Test 7a: Quiet audio should not be flagged as raised voice."""
        with patch(
            "videosearch.audio_analyzer.extract_audio_segment",
            side_effect=_mock_extract(quiet_wav),
        ):
            result = analyzer.analyze("video.mp4", 0.0, 2.0)

        # Default has_raised_voice is False (set by caller after all chunks)
        assert result["has_raised_voice"] is False

    def test_detect_raised_voice_with_outlier(self):
        """Test 9: detect_raised_voice flags chunks with RMS > mean + 2*stddev."""
        # Normal chunks at ~0.1 RMS, one outlier at 0.8
        video_rms_values = [0.10, 0.11, 0.09, 0.12, 0.10, 0.80]
        # The outlier (0.80) should be detected
        assert LibrosaAudioAnalyzer.detect_raised_voice(
            chunk_rms_max=0.80,
            video_rms_values=video_rms_values,
            stddev_threshold=2.0,
        ) is True

    def test_detect_raised_voice_normal_chunk(self):
        """Normal chunk should not be flagged."""
        video_rms_values = [0.10, 0.11, 0.09, 0.12, 0.10]
        assert LibrosaAudioAnalyzer.detect_raised_voice(
            chunk_rms_max=0.10,
            video_rms_values=video_rms_values,
            stddev_threshold=2.0,
        ) is False


class TestTempFileCleanup:
    """Test 8: Temp audio file cleaned up even on failure."""

    def test_cleanup_on_success(self, analyzer, loud_wav):
        created_temps = []

        def tracking_extract(video_path, start, end, ffmpeg_path="ffmpeg"):
            path = _mock_extract(loud_wav)(video_path, start, end, ffmpeg_path)
            created_temps.append(path)
            return path

        with patch(
            "videosearch.audio_analyzer.extract_audio_segment",
            side_effect=tracking_extract,
        ):
            analyzer.analyze("video.mp4", 0.0, 2.0)

        assert len(created_temps) == 1
        assert not Path(created_temps[0]).exists(), "Temp file should be cleaned up"

    def test_cleanup_on_failure(self, tmp_path):
        dummy_wav = tmp_path / "fail_cleanup.wav"
        _write_wav(str(dummy_wav), np.zeros(SAMPLE_RATE))

        created_temps = []

        def tracking_extract(video_path, start, end, ffmpeg_path="ffmpeg"):
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            import shutil
            shutil.copy2(str(dummy_wav), tmp)
            created_temps.append(tmp)
            return tmp

        analyzer = LibrosaAudioAnalyzer()
        with patch(
            "videosearch.audio_analyzer.extract_audio_segment",
            side_effect=tracking_extract,
        ), patch("videosearch.audio_analyzer.librosa") as mock_librosa:
            mock_librosa.load.side_effect = RuntimeError("librosa failed")
            with pytest.raises(RuntimeError, match="librosa failed"):
                analyzer.analyze("video.mp4", 0.0, 2.0)

        assert len(created_temps) == 1
        assert not Path(created_temps[0]).exists(), "Temp file should be cleaned up on error"
