"""Tests for ChunkMetadata data model and extraction sub-models."""

from videosearch.models import (
    AudioFeatures,
    ChunkMetadata,
    OCRResult,
    TranscriptSegment,
)


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
    # Core fields always present
    assert "video_id" in data
    assert "chunk_index" in data
    assert "start_time" in data
    assert "end_time" in data
    assert "duration" in data
    assert "scene_type" in data


# --- New extraction sub-model tests ---


def test_transcript_segment_instantiation():
    """TranscriptSegment instantiates with all fields."""
    segment = TranscriptSegment(
        text="hello",
        start=0.0,
        end=1.0,
        avg_logprob=-0.5,
        words=[{"word": "hello", "start": 0.0, "end": 1.0, "probability": 0.9}],
    )
    assert segment.text == "hello"
    assert segment.start == 0.0
    assert segment.end == 1.0
    assert segment.avg_logprob == -0.5
    assert len(segment.words) == 1
    assert segment.words[0]["word"] == "hello"
    assert segment.words[0]["probability"] == 0.9


def test_audio_features_instantiation():
    """AudioFeatures instantiates with all 10 fields including Optional pitch."""
    features = AudioFeatures(
        rms_mean=0.05,
        rms_max=0.2,
        rms_stddev=0.03,
        pitch_mean=None,
        pitch_max=None,
        pitch_stddev=None,
        zcr_mean=0.1,
        zcr_max=0.3,
        zcr_stddev=0.05,
        has_raised_voice=False,
    )
    assert features.rms_mean == 0.05
    assert features.rms_max == 0.2
    assert features.rms_stddev == 0.03
    assert features.pitch_mean is None
    assert features.pitch_max is None
    assert features.pitch_stddev is None
    assert features.zcr_mean == 0.1
    assert features.zcr_max == 0.3
    assert features.zcr_stddev == 0.05
    assert features.has_raised_voice is False


def test_ocr_result_instantiation():
    """OCRResult instantiates with text, confidence, timestamps, and bbox."""
    result = OCRResult(
        text="ABC123",
        confidence=0.95,
        first_seen=2.0,
        last_seen=6.0,
        bbox=[[0, 0], [100, 0], [100, 30], [0, 30]],
    )
    assert result.text == "ABC123"
    assert result.confidence == 0.95
    assert result.first_seen == 2.0
    assert result.last_seen == 6.0
    assert len(result.bbox) == 4
    assert result.bbox[0] == [0, 0]


def test_chunk_metadata_with_extraction_fields_roundtrip():
    """ChunkMetadata with all extraction fields serializes/deserializes identically."""
    segment = TranscriptSegment(
        text="stop right there",
        start=1.0,
        end=2.5,
        avg_logprob=-0.3,
        words=[
            {"word": "stop", "start": 1.0, "end": 1.3, "probability": 0.95},
            {"word": "right", "start": 1.3, "end": 1.6, "probability": 0.92},
            {"word": "there", "start": 1.6, "end": 2.5, "probability": 0.88},
        ],
    )
    features = AudioFeatures(
        rms_mean=0.08,
        rms_max=0.35,
        rms_stddev=0.06,
        pitch_mean=220.0,
        pitch_max=440.0,
        pitch_stddev=50.0,
        zcr_mean=0.12,
        zcr_max=0.4,
        zcr_stddev=0.07,
        has_raised_voice=True,
    )
    ocr = OCRResult(
        text="POLICE",
        confidence=0.98,
        first_seen=3.0,
        last_seen=8.0,
        bbox=[[10, 10], [200, 10], [200, 50], [10, 50]],
    )
    chunk = ChunkMetadata(
        video_id="bwc_003",
        chunk_index=2,
        start_time=10.0,
        end_time=40.0,
        duration=30.0,
        scene_type="detected",
        transcript=[segment],
        audio_features=features,
        ocr_results=[ocr],
    )
    json_str = chunk.model_dump_json()
    restored = ChunkMetadata.model_validate_json(json_str)
    assert restored == chunk
    assert restored.transcript[0].text == "stop right there"
    assert restored.audio_features.has_raised_voice is True
    assert restored.ocr_results[0].text == "POLICE"


def test_chunk_metadata_backward_compat_none_fields():
    """ChunkMetadata with extraction fields as None (backward compat) works."""
    chunk = ChunkMetadata(
        video_id="bwc_004",
        chunk_index=0,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        scene_type="sliding_window",
        transcript=None,
        audio_features=None,
        ocr_results=None,
    )
    assert chunk.transcript is None
    assert chunk.audio_features is None
    assert chunk.ocr_results is None

    # Roundtrip with None fields
    json_str = chunk.model_dump_json()
    restored = ChunkMetadata.model_validate_json(json_str)
    assert restored == chunk
    assert restored.transcript is None
