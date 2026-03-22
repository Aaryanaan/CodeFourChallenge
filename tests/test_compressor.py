"""Tests for FFmpegCompressor -- 720p CRF 28 h264 compression."""

import subprocess
from pathlib import Path

import pytest

from videosearch.compressor import FFmpegCompressor
from videosearch.protocols import Compressor


class TestCompressorProtocol:
    """Verify FFmpegCompressor satisfies the Compressor protocol."""

    def test_compressor_satisfies_protocol(self):
        compressor = FFmpegCompressor()
        assert isinstance(compressor, Compressor)

    def test_compressor_has_compress_method(self):
        compressor = FFmpegCompressor()
        assert hasattr(compressor, "compress")
        assert callable(compressor.compress)


class TestCompression:
    """Integration tests for video compression."""

    def test_compress_produces_720p(self, test_video, tmp_path):
        """Compress a 1080p video and verify output is 720p via ffprobe."""
        compressor = FFmpegCompressor()
        output_path = str(tmp_path / "compressed.mp4")
        compressor.compress(test_video, output_path)

        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "csv=p=0",
             output_path],
            capture_output=True, text=True, check=True,
        )
        height = int(result.stdout.strip())
        assert height == 720

    def test_compress_output_exists(self, test_video, tmp_path):
        """Verify output file exists after compression."""
        compressor = FFmpegCompressor()
        output_path = str(tmp_path / "compressed.mp4")
        compressor.compress(test_video, output_path)
        assert Path(output_path).exists()

    def test_compress_output_smaller(self, test_video, tmp_path):
        """Verify compressed output is smaller than 1080p input."""
        compressor = FFmpegCompressor()
        output_path = str(tmp_path / "compressed.mp4")
        compressor.compress(test_video, output_path)

        input_size = Path(test_video).stat().st_size
        output_size = Path(output_path).stat().st_size
        assert output_size < input_size

    def test_compress_returns_path(self, test_video, tmp_path):
        """Verify compress() returns the output_path string."""
        compressor = FFmpegCompressor()
        output_path = str(tmp_path / "compressed.mp4")
        result = compressor.compress(test_video, output_path)
        assert result == output_path

    def test_compress_nonexistent_input(self, tmp_path):
        """Verify FileNotFoundError on nonexistent input file."""
        compressor = FFmpegCompressor()
        output_path = str(tmp_path / "compressed.mp4")
        with pytest.raises(FileNotFoundError):
            compressor.compress("/nonexistent/video.mp4", output_path)
