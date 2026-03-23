import os
import subprocess

import cv2
import numpy as np
import pytest
from pathlib import Path

from videosearch.audio_utils import extract_audio_segment


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


@pytest.fixture(scope="session")
def test_audio_wav(test_video):
    """Extract a mono 16kHz WAV from first 5s of test_video.

    Session-scoped for reuse across tests. Cleans up in finalizer.
    """
    wav_path = extract_audio_segment(test_video, 0.0, 5.0)
    yield wav_path
    os.unlink(wav_path)


@pytest.fixture(scope="session")
def test_frame_with_text(tmp_path_factory):
    """Create a 640x480 white PNG image with black text 'TEST123'.

    Uses OpenCV cv2.putText for text rendering.
    """
    frame_dir = tmp_path_factory.mktemp("frames")
    frame_path = frame_dir / "text_frame.png"
    # White background
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    # Black text
    cv2.putText(
        img, "TEST123",
        (100, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (0, 0, 0), 3,
    )
    cv2.imwrite(str(frame_path), img)
    return str(frame_path)


@pytest.fixture(scope="session")
def test_frame_no_text(tmp_path_factory):
    """Create a 640x480 solid black PNG image (no text)."""
    frame_dir = tmp_path_factory.mktemp("frames_blank")
    frame_path = frame_dir / "blank_frame.png"
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(frame_path), img)
    return str(frame_path)
