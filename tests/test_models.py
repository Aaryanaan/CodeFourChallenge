"""Tests for ChunkMetadata data model."""

from videosearch.models import ChunkMetadata


def test_chunk_metadata_instantiation():
    """ChunkMetadata can be created with all required fields."""
    chunk = ChunkMetadata(
        video_id="test_video",
        chunk_index=0,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        scene_type="detected",
    )
    assert chunk.video_id == "test_video"
    assert chunk.chunk_index == 0
    assert chunk.start_time == 0.0
    assert chunk.end_time == 30.0
    assert chunk.duration == 30.0
    assert chunk.scene_type == "detected"


def test_chunk_metadata_sliding_window_type():
    """ChunkMetadata accepts 'sliding_window' as scene_type."""
    chunk = ChunkMetadata(
        video_id="bwc_001",
        chunk_index=5,
        start_time=120.0,
        end_time=150.0,
        duration=30.0,
        scene_type="sliding_window",
    )
    assert chunk.scene_type == "sliding_window"


def test_chunk_metadata_json_roundtrip():
    """ChunkMetadata serializes to JSON and deserializes back identically."""
    chunk = ChunkMetadata(
        video_id="bwc_002",
        chunk_index=3,
        start_time=60.0,
        end_time=90.5,
        duration=30.5,
        scene_type="detected",
    )
    json_str = chunk.model_dump_json()
    restored = ChunkMetadata.model_validate_json(json_str)
    assert restored == chunk
    assert restored.video_id == "bwc_002"
    assert restored.start_time == 60.0
    assert restored.end_time == 90.5


def test_chunk_metadata_model_dump():
    """ChunkMetadata model_dump returns a dict with all fields."""
    chunk = ChunkMetadata(
        video_id="test",
        chunk_index=0,
        start_time=0.0,
        end_time=10.0,
        duration=10.0,
        scene_type="detected",
    )
    data = chunk.model_dump()
    assert isinstance(data, dict)
    # Check core keys are present (optional extraction fields may also exist)
    core_keys = {"video_id", "chunk_index", "start_time", "end_time", "duration", "scene_type"}
    assert core_keys <= set(data.keys())
