"""Tests for MetadataWriter JSON serialization and deserialization."""

import json

import pytest

from videosearch.metadata_writer import MetadataWriter
from videosearch.models import (
    AudioFeatures,
    ChunkMetadata,
    OCRResult,
    TranscriptSegment,
)


def _make_basic_chunk(chunk_index: int = 0) -> ChunkMetadata:
    """Create a ChunkMetadata with only required fields (no extraction data)."""
    return ChunkMetadata(
        video_id="test_video",
        chunk_index=chunk_index,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        scene_type="detected",
    )


def _make_enriched_chunk(chunk_index: int = 0) -> ChunkMetadata:
    """Create a ChunkMetadata with all extraction fields populated."""
    return ChunkMetadata(
        video_id="test_video",
        chunk_index=chunk_index,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        scene_type="detected",
        transcript=[
            TranscriptSegment(
                text="Hello world",
                start=1.0,
                end=2.5,
                avg_logprob=-0.25,
                words=[
                    {"word": "Hello", "start": 1.0, "end": 1.5, "probability": 0.95},
                    {"word": "world", "start": 1.6, "end": 2.5, "probability": 0.92},
                ],
            ),
        ],
        audio_features=AudioFeatures(
            rms_mean=0.05,
            rms_max=0.15,
            rms_stddev=0.02,
            pitch_mean=220.0,
            pitch_max=440.0,
            pitch_stddev=50.0,
            zcr_mean=0.1,
            zcr_max=0.3,
            zcr_stddev=0.05,
            has_raised_voice=False,
        ),
        ocr_results=[
            OCRResult(
                text="STOP",
                confidence=0.95,
                first_seen=2.0,
                last_seen=6.0,
                bbox=[[0, 0], [80, 0], [80, 25], [0, 25]],
            ),
        ],
    )


class TestMetadataWriterWrite:
    """MetadataWriter.write() creates JSON files."""

    def test_write_creates_json_file(self, tmp_metadata_dir):
        writer = MetadataWriter(metadata_dir=tmp_metadata_dir)
        chunks = [_make_basic_chunk()]
        path = writer.write("test_video", chunks)
        assert path.exists()
        assert path.name == "test_video.json"

    def test_write_creates_list_of_serialized_chunks(self, tmp_metadata_dir):
        writer = MetadataWriter(metadata_dir=tmp_metadata_dir)
        chunks = [_make_basic_chunk(0), _make_basic_chunk(1)]
        path = writer.write("test_video", chunks)
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["video_id"] == "test_video"
        assert data[1]["chunk_index"] == 1

    def test_write_creates_dir_if_not_exists(self, tmp_path):
        new_dir = tmp_path / "new_metadata"
        assert not new_dir.exists()
        writer = MetadataWriter(metadata_dir=new_dir)
        chunks = [_make_basic_chunk()]
        path = writer.write("test_video", chunks)
        assert new_dir.exists()
        assert path.exists()

    def test_write_enriched_chunk_serializes_correctly(self, tmp_metadata_dir):
        writer = MetadataWriter(metadata_dir=tmp_metadata_dir)
        chunks = [_make_enriched_chunk()]
        path = writer.write("test_video", chunks)
        data = json.loads(path.read_text())
        assert data[0]["transcript"] is not None
        assert data[0]["transcript"][0]["text"] == "Hello world"
        assert data[0]["audio_features"]["rms_mean"] == 0.05
        assert data[0]["ocr_results"][0]["text"] == "STOP"

    def test_write_none_extraction_fields(self, tmp_metadata_dir):
        """ChunkMetadata with None extraction fields serializes correctly."""
        writer = MetadataWriter(metadata_dir=tmp_metadata_dir)
        chunks = [_make_basic_chunk()]
        path = writer.write("test_video", chunks)
        data = json.loads(path.read_text())
        assert data[0]["transcript"] is None
        assert data[0]["audio_features"] is None
        assert data[0]["ocr_results"] is None


class TestMetadataWriterLoad:
    """MetadataWriter.load() reads JSON files back to ChunkMetadata."""

    def test_load_returns_list_of_chunk_metadata(self, tmp_metadata_dir):
        writer = MetadataWriter(metadata_dir=tmp_metadata_dir)
        original = [_make_basic_chunk(0), _make_basic_chunk(1)]
        writer.write("test_video", original)
        loaded = writer.load("test_video")
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert all(isinstance(c, ChunkMetadata) for c in loaded)

    def test_load_nonexistent_raises_file_not_found(self, tmp_metadata_dir):
        writer = MetadataWriter(metadata_dir=tmp_metadata_dir)
        with pytest.raises(FileNotFoundError):
            writer.load("nonexistent_video")

    def test_load_phase1_style_no_extraction_fields(self, tmp_metadata_dir):
        """Loading JSON without extraction fields returns ChunkMetadata with None."""
        # Simulate Phase 1 JSON (no transcript, audio_features, ocr_results)
        data = [
            {
                "video_id": "old_vid",
                "chunk_index": 0,
                "start_time": 0.0,
                "end_time": 30.0,
                "duration": 30.0,
                "scene_type": "detected",
            }
        ]
        json_path = tmp_metadata_dir / "old_vid.json"
        json_path.write_text(json.dumps(data))

        writer = MetadataWriter(metadata_dir=tmp_metadata_dir)
        loaded = writer.load("old_vid")
        assert len(loaded) == 1
        assert loaded[0].transcript is None
        assert loaded[0].audio_features is None
        assert loaded[0].ocr_results is None


class TestMetadataWriterRoundTrip:
    """Round-trip: write then load returns identical ChunkMetadata."""

    def test_write_and_load(self, tmp_metadata_dir):
        writer = MetadataWriter(metadata_dir=tmp_metadata_dir)
        original = [_make_enriched_chunk(0), _make_basic_chunk(1)]
        writer.write("test_video", original)
        loaded = writer.load("test_video")

        assert len(loaded) == len(original)
        for orig, load in zip(original, loaded):
            assert orig.video_id == load.video_id
            assert orig.chunk_index == load.chunk_index
            assert orig.start_time == load.start_time
            assert orig.end_time == load.end_time
            assert orig.duration == load.duration
            assert orig.scene_type == load.scene_type

        # Check enriched chunk
        assert loaded[0].transcript is not None
        assert loaded[0].transcript[0].text == "Hello world"
        assert loaded[0].audio_features is not None
        assert loaded[0].audio_features.rms_mean == 0.05
        assert loaded[0].audio_features.has_raised_voice is False
        assert loaded[0].ocr_results is not None
        assert loaded[0].ocr_results[0].text == "STOP"
        assert loaded[0].ocr_results[0].confidence == 0.95

        # Check basic chunk
        assert loaded[1].transcript is None
        assert loaded[1].audio_features is None
        assert loaded[1].ocr_results is None

    def test_round_trip_preserves_pydantic_model_equality(self, tmp_metadata_dir):
        """Pydantic model equality check after round-trip."""
        writer = MetadataWriter(metadata_dir=tmp_metadata_dir)
        original = [_make_enriched_chunk()]
        writer.write("test_video", original)
        loaded = writer.load("test_video")
        assert original == loaded
