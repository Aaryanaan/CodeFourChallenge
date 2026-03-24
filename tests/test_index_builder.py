"""Tests for IndexBuilder orchestrator and helper functions."""

import pytest

from videosearch.models import (
    AudioFeatures,
    ChunkMetadata,
    OCRResult,
    TranscriptSegment,
)


def _make_audio(rms_mean: float = 0.1) -> AudioFeatures:
    return AudioFeatures(
        rms_mean=rms_mean,
        rms_max=rms_mean * 2,
        rms_stddev=0.01,
        zcr_mean=0.05,
        zcr_max=0.1,
        zcr_stddev=0.01,
        has_raised_voice=False,
    )


def _make_chunk(
    video_id: str = "v1",
    chunk_index: int = 0,
    transcript_texts: list[str] | None = None,
    ocr_texts: list[str] | None = None,
    rms_mean: float = 0.1,
    has_audio: bool = True,
) -> ChunkMetadata:
    transcript = None
    if transcript_texts is not None:
        transcript = [
            TranscriptSegment(
                text=t, start=0.0, end=1.0, avg_logprob=-0.3, words=[]
            )
            for t in transcript_texts
        ]
    ocr_results = None
    if ocr_texts is not None:
        ocr_results = [
            OCRResult(
                text=t,
                confidence=0.95,
                first_seen=0.0,
                last_seen=1.0,
                bbox=[[0, 0], [1, 0], [1, 1], [0, 1]],
            )
            for t in ocr_texts
        ]
    audio = _make_audio(rms_mean) if has_audio else None
    return ChunkMetadata(
        video_id=video_id,
        chunk_index=chunk_index,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        scene_type="detected",
        transcript=transcript,
        audio_features=audio,
        ocr_results=ocr_results,
    )


# --- build_combined_text tests ---

class TestBuildCombinedText:
    def test_combined_text_format(self):
        """Chunk with both transcript and OCR returns labeled format."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk(transcript_texts=["hello", "world"], ocr_texts=["STOP SIGN"])
        result = build_combined_text(chunk)
        assert result == "Transcript: hello world\nOCR: STOP SIGN"

    def test_combined_text_transcript_only(self):
        """Chunk with only transcript returns transcript section."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk(transcript_texts=["hello", "world"])
        result = build_combined_text(chunk)
        assert result == "Transcript: hello world"

    def test_combined_text_ocr_only(self):
        """Chunk with only OCR returns OCR section."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk(ocr_texts=["STOP"])
        result = build_combined_text(chunk)
        assert result == "OCR: STOP"

    def test_combined_text_empty(self):
        """Chunk with neither transcript nor OCR returns empty string."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk()
        result = build_combined_text(chunk)
        assert result == ""


# --- skip empty chunks test ---

class TestSkipEmptyChunks:
    def test_skip_empty_chunks(self):
        """Chunks with no transcript AND no OCR are excluded from embeddable set."""
        from videosearch.index_builder import build_combined_text

        chunks = [
            _make_chunk(chunk_index=0, transcript_texts=["hello"]),
            _make_chunk(chunk_index=1),  # empty — no transcript, no OCR
            _make_chunk(chunk_index=2, ocr_texts=["EXIT"]),
        ]
        embeddable = [c for c in chunks if build_combined_text(c)]
        assert len(embeddable) == 2
        assert embeddable[0].chunk_index == 0
        assert embeddable[1].chunk_index == 2


# --- compute_volume_level tests ---

class TestComputeVolumeLevel:
    def test_volume_level_bins(self):
        """Volume level is quiet/normal/loud based on RMS distribution."""
        from videosearch.index_builder import compute_volume_level

        # Create a distribution with tight clustering around 0.1 so outliers
        # are clearly beyond 2 stddevs. 100 chunks at 0.1 gives mean~0.1,
        # stddev~0.0 so 0.01 is clearly quiet and 0.5 is clearly loud.
        chunks = [
            _make_chunk(chunk_index=i, rms_mean=0.1) for i in range(100)
        ]
        quiet_chunk = _make_chunk(chunk_index=100, rms_mean=0.01)
        loud_chunk = _make_chunk(chunk_index=101, rms_mean=0.5)
        all_chunks = chunks + [quiet_chunk, loud_chunk]

        assert compute_volume_level(quiet_chunk, all_chunks) == "quiet"
        assert compute_volume_level(loud_chunk, all_chunks) == "loud"
        assert compute_volume_level(chunks[0], all_chunks) == "normal"


# --- boolean flag tests ---

class TestBooleanFlags:
    def test_has_speech_flag(self):
        """Chunk with transcript has truthy transcript field."""
        chunk_with = _make_chunk(transcript_texts=["hello"])
        chunk_without = _make_chunk()
        assert bool(chunk_with.transcript)
        assert not chunk_without.transcript

    def test_has_ocr_flag(self):
        """Chunk with OCR results has truthy ocr_results field."""
        chunk_with = _make_chunk(ocr_texts=["STOP"])
        chunk_without = _make_chunk()
        assert bool(chunk_with.ocr_results)
        assert not chunk_without.ocr_results
