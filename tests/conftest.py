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
