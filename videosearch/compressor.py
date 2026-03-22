"""FFmpeg-based video compression.

Compresses video to 720p CRF 28 h264 using ffmpeg subprocess.
Satisfies the Compressor protocol structurally (no inheritance).
"""

import shutil
import subprocess
from pathlib import Path


class FFmpegCompressor:
    """Compress video to 720p using ffmpeg subprocess.

    Uses h264 codec with CRF 28 quality and scale=-2:720 for even-dimension
    safe downscaling. Validates ffmpeg availability on init.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        if not shutil.which(self.ffmpeg_path):
            raise FileNotFoundError(
                f"ffmpeg not found at: {self.ffmpeg_path}. "
                f"Install with: brew install ffmpeg"
            )

    def compress(self, video_path: str, output_path: str) -> str:
        """Compress video to 720p CRF 28 h264. Returns output path."""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cmd = [
            self.ffmpeg_path, "-i", video_path,
            "-vf", "scale=-2:720",       # 720p, -2 ensures even dimensions
            "-c:v", "libx264",            # H.264 codec
            "-crf", "28",                 # Quality level
            "-preset", "fast",            # Encoding speed
            "-c:a", "aac",               # Audio codec
            "-b:a", "128k",              # Audio bitrate
            "-y",                         # Overwrite output
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
