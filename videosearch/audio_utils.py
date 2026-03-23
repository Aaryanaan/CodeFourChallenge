"""Shared audio extraction utility using ffmpeg.

Extracts mono 16kHz WAV audio segments from video files for use by
Transcriber and AudioAnalyzer. Caller is responsible for cleanup
of the temporary file via try/finally.
"""

import subprocess
import tempfile
from pathlib import Path


def extract_audio_segment(
    video_path: str,
    start: float,
    end: float,
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
) -> str:
    """Extract an audio segment from a video file as mono WAV.

    Args:
        video_path: Path to the source video file.
        start: Start time in seconds.
        end: End time in seconds.
        sample_rate: Output sample rate in Hz (default 16000).
        ffmpeg_path: Path to ffmpeg binary.

    Returns:
        Path to the temporary WAV file. Caller must clean up via os.unlink().

    Raises:
        FileNotFoundError: If video_path does not exist.
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    duration = end - start

    # Create temp file for output (caller responsible for cleanup)
    fd, output_path = tempfile.mkstemp(suffix=".wav")
    # Close the file descriptor; ffmpeg will write to the path
    import os
    os.close(fd)

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", video_path,
        "-ss", str(start),
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    return output_path
