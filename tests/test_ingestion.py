"""Tests for IngestionPipeline orchestrator.

All external components (FFmpegCompressor, SceneAwareChunker, WhisperTranscriber,
LibrosaAudioAnalyzer, PaddleOCRExtractor, MetadataWriter) are mocked so tests run
without heavy dependencies or real video files.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from videosearch.models import (
    AudioFeatures,
    ChunkMetadata,
    OCRResult,
    TranscriptSegment,
)

_CAPTION_RESULT = {"caption": "test caption", "cached": False}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path):
    """Return a mock Settings object with all paths pointing to tmp_path."""
    s = MagicMock()
    s.video_dir = tmp_path / "videos"
    s.metadata_dir = tmp_path / "metadata"
    s.ffmpeg_path = "ffmpeg"
    s.pyscenedetect_threshold = 40.0
    s.chunk_min_duration = 10.0
    s.chunk_max_duration = 60.0
    s.sliding_window_size = 30.0
    s.sliding_window_overlap = 10.0
    s.whisper_model = "large-v3"
    s.whisper_compute_type = "auto"
    s.ocr_confidence_threshold = 0.7
    s.raised_voice_stddev_threshold = 2.0
    s.ocr_frame_interval = 2.0
    return s


def _make_chunks(video_id: str, n: int = 2) -> list[ChunkMetadata]:
    """Return a list of n ChunkMetadata objects for video_id."""
    chunks = []
    for i in range(n):
        start = float(i * 30)
        end = float((i + 1) * 30)
        chunks.append(
            ChunkMetadata(
                video_id=video_id,
                chunk_index=i,
                start_time=start,
                end_time=end,
                duration=end - start,
                scene_type="sliding_window",
            )
        )
    return chunks


# Reusable mock return values
_TRANSCRIBE_RESULT = {
    "segments": [
        {
            "text": "test",
            "start": 0.0,
            "end": 1.0,
            "avg_logprob": -0.3,
            "words": [],
        }
    ],
    "language": "en",
    "language_probability": 0.99,
}

_AUDIO_RESULT = {
    "rms_mean": 0.1,
    "rms_max": 0.3,
    "rms_stddev": 0.05,
    "pitch_mean": 200.0,
    "pitch_max": 400.0,
    "pitch_stddev": 50.0,
    "zcr_mean": 0.05,
    "zcr_max": 0.1,
    "zcr_stddev": 0.02,
    "has_raised_voice": False,
}

_OCR_RESULT = {
    "results": [
        {
            "text": "POLICE",
            "confidence": 0.95,
            "first_seen": 1.0,
            "last_seen": 3.0,
            "bbox": [[0, 0], [100, 0], [100, 30], [0, 30]],
        }
    ],
    "frame_count": 5,
    "backend": "paddleocr",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_components():
    """Patch all external components used by IngestionPipeline."""
    with (
        patch("videosearch.ingestion.FFmpegCompressor") as MockCompressor,
        patch("videosearch.ingestion.SceneAwareChunker") as MockChunker,
        patch("videosearch.ingestion.WhisperTranscriber") as MockTranscriber,
        patch("videosearch.ingestion.LibrosaAudioAnalyzer") as MockAudioAnalyzer,
        patch("videosearch.ingestion.PaddleOCRExtractor") as MockOCR,
        patch("videosearch.ingestion.MetadataWriter") as MockWriter,
        patch("videosearch.ingestion.GeminiCaptioner") as MockCaptioner,
    ):
        yield {
            "Compressor": MockCompressor,
            "Chunker": MockChunker,
            "Transcriber": MockTranscriber,
            "AudioAnalyzer": MockAudioAnalyzer,
            "OCR": MockOCR,
            "Writer": MockWriter,
            "Captioner": MockCaptioner,
        }


def _build_pipeline(
    tmp_path, mock_components, video_id="bodycam_001", n_chunks=2,
    include_captioner=False,
):
    """Create an IngestionPipeline with all components mocked and pre-configured."""
    from videosearch.ingestion import IngestionPipeline

    settings = _make_settings(tmp_path)
    chunks = _make_chunks(video_id, n=n_chunks)

    # Configure instances
    mock_compressor = mock_components["Compressor"].return_value
    compressed_path = str(
        settings.video_dir / "compressed" / f"{video_id}_720p.mp4"
    )
    mock_compressor.compress.return_value = compressed_path

    mock_chunker = mock_components["Chunker"].return_value
    mock_chunker.chunk.return_value = chunks

    mock_transcriber = mock_components["Transcriber"].return_value
    mock_transcriber.transcribe.return_value = _TRANSCRIBE_RESULT

    mock_audio = mock_components["AudioAnalyzer"].return_value
    mock_audio.analyze.return_value = _AUDIO_RESULT

    mock_ocr = mock_components["OCR"].return_value
    mock_ocr.extract.return_value = _OCR_RESULT

    if include_captioner:
        mock_captioner = mock_components["Captioner"].return_value
        mock_captioner.caption.return_value = _CAPTION_RESULT
        settings.caption_cache_dir = tmp_path / "cache" / "captions"
        settings.caption_cost_per_chunk = 0.003

    pipeline = IngestionPipeline(settings)
    return pipeline, settings, chunks, compressed_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ingest_calls_compress(tmp_path, mock_components):
    """ingest() calls compressor.compress with (video_path, expected_output_path)."""
    pipeline, settings, _chunks, compressed_path = _build_pipeline(
        tmp_path, mock_components
    )
    video_path = "/path/to/bodycam_001.mp4"

    pipeline.ingest(video_path)

    mock_compressor = mock_components["Compressor"].return_value
    mock_compressor.compress.assert_called_once_with(video_path, compressed_path)


def test_ingest_calls_chunk(tmp_path, mock_components):
    """ingest() calls chunker.chunk(compressed_path) after compression."""
    pipeline, settings, _chunks, compressed_path = _build_pipeline(
        tmp_path, mock_components
    )
    video_path = "/path/to/bodycam_001.mp4"

    pipeline.ingest(video_path)

    mock_chunker = mock_components["Chunker"].return_value
    mock_chunker.chunk.assert_called_once_with(compressed_path)


def test_ingest_runs_extractors_per_chunk(tmp_path, mock_components):
    """For 2 chunks, each extractor is called 2 times with correct args."""
    pipeline, settings, chunks, compressed_path = _build_pipeline(
        tmp_path, mock_components, n_chunks=2
    )
    video_path = "/path/to/bodycam_001.mp4"

    pipeline.ingest(video_path)

    mock_transcriber = mock_components["Transcriber"].return_value
    mock_audio = mock_components["AudioAnalyzer"].return_value
    mock_ocr = mock_components["OCR"].return_value

    assert mock_transcriber.transcribe.call_count == 2
    assert mock_audio.analyze.call_count == 2
    assert mock_ocr.extract.call_count == 2

    # Verify each call uses compressed_path and the correct chunk time window
    for chunk in chunks:
        mock_transcriber.transcribe.assert_any_call(
            compressed_path, chunk.start_time, chunk.end_time
        )
        mock_audio.analyze.assert_any_call(
            compressed_path, chunk.start_time, chunk.end_time
        )
        mock_ocr.extract.assert_any_call(
            compressed_path, chunk.start_time, chunk.end_time
        )


def test_ingest_two_pass_raised_voice(tmp_path, mock_components):
    """After extraction, detect_raised_voice is called per chunk with full rms list."""
    pipeline, settings, _chunks, compressed_path = _build_pipeline(
        tmp_path, mock_components, n_chunks=2
    )
    video_path = "/path/to/bodycam_001.mp4"

    with patch(
        "videosearch.ingestion.LibrosaAudioAnalyzer.detect_raised_voice",
        return_value=True,
    ) as mock_detect:
        pipeline.ingest(video_path)

    # detect_raised_voice called once per chunk
    assert mock_detect.call_count == 2

    # Each call receives the chunk's rms_max and the full list of all rms_max values
    expected_rms_values = [_AUDIO_RESULT["rms_max"], _AUDIO_RESULT["rms_max"]]
    for call in mock_detect.call_args_list:
        args, kwargs = call
        assert args[0] == _AUDIO_RESULT["rms_max"]
        assert args[1] == expected_rms_values
        assert args[2] == settings.raised_voice_stddev_threshold


def test_ingest_writes_metadata(tmp_path, mock_components):
    """metadata_writer.write(video_id, chunks) is called after extraction."""
    pipeline, settings, chunks, _compressed = _build_pipeline(
        tmp_path, mock_components
    )
    video_path = "/path/to/bodycam_001.mp4"

    pipeline.ingest(video_path)

    mock_writer = mock_components["Writer"].return_value
    mock_writer.write.assert_called_once()
    call_args = mock_writer.write.call_args
    assert call_args[0][0] == "bodycam_001"  # video_id
    written_chunks = call_args[0][1]
    assert len(written_chunks) == 2


def test_ingest_returns_video_id(tmp_path, mock_components):
    """ingest('/path/to/bodycam_001.mp4') returns 'bodycam_001'."""
    pipeline, _settings, _chunks, _compressed = _build_pipeline(
        tmp_path, mock_components
    )
    result = pipeline.ingest("/path/to/bodycam_001.mp4")
    assert result == "bodycam_001"


def test_ingest_extractor_failure_continues(tmp_path, mock_components):
    """If transcriber raises, chunk.transcript is None but audio and OCR still run."""
    pipeline, settings, _chunks, compressed_path = _build_pipeline(
        tmp_path, mock_components, n_chunks=2
    )
    mock_transcriber = mock_components["Transcriber"].return_value
    mock_transcriber.transcribe.side_effect = RuntimeError("Whisper exploded")

    # Should not raise
    pipeline.ingest("/path/to/bodycam_001.mp4")

    # Audio and OCR still ran for both chunks
    mock_audio = mock_components["AudioAnalyzer"].return_value
    mock_ocr = mock_components["OCR"].return_value
    assert mock_audio.analyze.call_count == 2
    assert mock_ocr.extract.call_count == 2


def test_ingest_populates_chunk_fields(tmp_path, mock_components):
    """After extraction, chunk fields are populated as proper model instances."""
    pipeline, settings, chunks, _compressed = _build_pipeline(
        tmp_path, mock_components, n_chunks=1
    )

    pipeline.ingest("/path/to/bodycam_001.mp4")

    chunk = chunks[0]
    assert chunk.transcript is not None
    assert isinstance(chunk.transcript, list)
    assert len(chunk.transcript) == 1
    assert isinstance(chunk.transcript[0], TranscriptSegment)

    assert chunk.audio_features is not None
    assert isinstance(chunk.audio_features, AudioFeatures)

    assert chunk.ocr_results is not None
    assert isinstance(chunk.ocr_results, list)
    assert len(chunk.ocr_results) == 1
    assert isinstance(chunk.ocr_results[0], OCRResult)


# ---------------------------------------------------------------------------
# Parallel extraction and --caption tests
# ---------------------------------------------------------------------------


def test_parallel_extraction_runs_all_extractors(tmp_path, mock_components):
    """With 1 chunk, all 3 extractors are called exactly once and fields populated."""
    pipeline, settings, chunks, compressed_path = _build_pipeline(
        tmp_path, mock_components, n_chunks=1
    )

    pipeline.ingest("/path/to/bodycam_001.mp4")

    chunk = chunks[0]
    mock_transcriber = mock_components["Transcriber"].return_value
    mock_audio = mock_components["AudioAnalyzer"].return_value
    mock_ocr = mock_components["OCR"].return_value

    assert mock_transcriber.transcribe.call_count == 1
    assert mock_audio.analyze.call_count == 1
    assert mock_ocr.extract.call_count == 1
    assert chunk.transcript is not None
    assert chunk.audio_features is not None
    assert chunk.ocr_results is not None


def test_parallel_extraction_with_caption(tmp_path, mock_components):
    """When include_caption=True with captioner injected, visual_caption populated."""
    pipeline, settings, chunks, compressed_path = _build_pipeline(
        tmp_path, mock_components, n_chunks=1, include_captioner=True
    )

    pipeline.ingest("/path/to/bodycam_001.mp4", include_caption=True)

    chunk = chunks[0]
    mock_captioner = mock_components["Captioner"].return_value
    assert mock_captioner.caption.call_count == 1
    assert chunk.visual_caption == "test caption"


def test_parallel_extraction_single_failure(tmp_path, mock_components):
    """When transcriber fails, audio and OCR still run and populate chunk fields."""
    pipeline, settings, chunks, compressed_path = _build_pipeline(
        tmp_path, mock_components, n_chunks=1
    )
    mock_transcriber = mock_components["Transcriber"].return_value
    mock_transcriber.transcribe.side_effect = RuntimeError("Whisper exploded")

    pipeline.ingest("/path/to/bodycam_001.mp4")

    chunk = chunks[0]
    assert chunk.transcript is None  # failed
    assert chunk.audio_features is not None  # still ran
    assert chunk.ocr_results is not None  # still ran


def test_parallel_extraction_all_fail(tmp_path, mock_components):
    """When all 3 extractors raise, ingest still completes without exception."""
    pipeline, settings, chunks, compressed_path = _build_pipeline(
        tmp_path, mock_components, n_chunks=1
    )
    mock_components["Transcriber"].return_value.transcribe.side_effect = RuntimeError("fail")
    mock_components["AudioAnalyzer"].return_value.analyze.side_effect = RuntimeError("fail")
    mock_components["OCR"].return_value.extract.side_effect = RuntimeError("fail")

    # Should NOT raise
    pipeline.ingest("/path/to/bodycam_001.mp4")

    # Metadata writer still called
    mock_writer = mock_components["Writer"].return_value
    mock_writer.write.assert_called_once()


def test_ingest_with_caption_flag(tmp_path, mock_components):
    """pipeline.ingest(path, include_caption=True) calls captioner for each chunk."""
    pipeline, settings, chunks, compressed_path = _build_pipeline(
        tmp_path, mock_components, n_chunks=2, include_captioner=True
    )

    pipeline.ingest("/path/to/bodycam_001.mp4", include_caption=True)

    mock_captioner = mock_components["Captioner"].return_value
    assert mock_captioner.caption.call_count == 2
