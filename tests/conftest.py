import subprocess

import pytest
from pathlib import Path


@pytest.fixture
def tmp_video_dir(tmp_path):
    d = tmp_path / "videos"
    d.mkdir()
    return d


@pytest.fixture
def tmp_metadata_dir(tmp_path):
    d = tmp_path / "metadata"
    d.mkdir()
    return d


@pytest.fixture(scope="session")
def test_video(tmp_path_factory):
    """Generate a 30s synthetic 1080p test video with scene changes."""
    video_dir = tmp_path_factory.mktemp("videos")
    video_path = video_dir / "test_input.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        "testsrc=duration=30:size=1920x1080:rate=30",
        "-f", "lavfi", "-i",
        "sine=frequency=440:duration=30",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(video_path)


@pytest.fixture(scope="session")
def test_video_long(tmp_path_factory):
    """Generate a 120s synthetic video for chunking tests.

    Uses color source changes to create detectable scene boundaries.
    Avoids drawtext filter (requires libfreetype which may not be available).
    """
    video_dir = tmp_path_factory.mktemp("videos_long")
    video_path = video_dir / "test_long.mp4"
    # Create video with abrupt color changes to simulate scene boundaries.
    # Pure color sources always available -- no drawtext filter needed.
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        "color=c=red:duration=25:size=1920x1080:rate=30",
        "-f", "lavfi", "-i",
        "color=c=blue:duration=35:size=1920x1080:rate=30",
        "-f", "lavfi", "-i",
        "color=c=green:duration=30:size=1920x1080:rate=30",
        "-f", "lavfi", "-i",
        "color=c=yellow:duration=30:size=1920x1080:rate=30",
        "-f", "lavfi", "-i",
        "sine=frequency=440:duration=120",
        "-filter_complex",
        "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[outv]",
        "-map", "[outv]", "-map", "4:a",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(video_path)
