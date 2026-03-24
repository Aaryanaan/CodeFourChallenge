"""Tests for IndexBuilder (IDX-01, IDX-04 integration)."""
import pytest
from unittest.mock import MagicMock, patch

from videosearch.index_builder import IndexBuilder, compute_volume_level, build_combined_text
from videosearch.models import (
    ChunkMetadata, TranscriptSegment, AudioFeatures, OCRResult,
)


def _make_chunk(
    video_id="v1", chunk_index=0,
    transcript_text=None, ocr_text=None,
    rms_mean=0.05, has_raised_voice=False,
):
    transcript = None
    if transcript_text:
        transcript = [TranscriptSegment(
            text=transcript_text, start=0.0, end=10.0,
            avg_logprob=-0.3,
            words=[{"word": w, "start": 0.0, "end": 1.0, "probability": 0.9}
                   for w in transcript_text.split()],
        )]
    ocr_results = None
    if ocr_text:
        ocr_results = [OCRResult(
            text=ocr_text, confidence=0.95,
            first_seen=0.0, last_seen=5.0,
            bbox=[[0, 0], [100, 0], [100, 30], [0, 30]],
        )]
    audio = AudioFeatures(
        rms_mean=rms_mean, rms_max=rms_mean * 2, rms_stddev=rms_mean * 0.3,
        zcr_mean=0.05, zcr_max=0.1, zcr_stddev=0.02,
        has_raised_voice=has_raised_voice,
    )
    return ChunkMetadata(
        video_id=video_id, chunk_index=chunk_index,
        start_time=0.0, end_time=30.0, duration=30.0,
        scene_type="detected", transcript=transcript,
        audio_features=audio, ocr_results=ocr_results,
    )


def test_combined_text_format():
    chunk = _make_chunk(transcript_text="hello world", ocr_text="STOP SIGN")
    text = build_combined_text(chunk)
    assert text == "Transcript: hello world\nOCR: STOP SIGN"


def test_combined_text_transcript_only():
    chunk = _make_chunk(transcript_text="hello world", ocr_text=None)
    text = build_combined_text(chunk)
    assert text == "Transcript: hello world"


def test_combined_text_ocr_only():
    chunk = _make_chunk(transcript_text=None, ocr_text="STOP")
    text = build_combined_text(chunk)
    assert text == "OCR: STOP"


def test_combined_text_empty():
    chunk = _make_chunk(transcript_text=None, ocr_text=None)
    text = build_combined_text(chunk)
    assert text == ""


def test_skip_empty_chunks():
    """Chunks with no transcript AND no OCR should not be embedded (D-03)."""
    chunks = [
        _make_chunk(chunk_index=0, transcript_text="hello"),
        _make_chunk(chunk_index=1),  # empty — no transcript, no OCR
        _make_chunk(chunk_index=2, ocr_text="STOP"),
    ]
    embeddable = [c for c in chunks if build_combined_text(c)]
    assert len(embeddable) == 2


def test_volume_level_bins():
    """Volume levels should be quiet/normal/loud based on per-video RMS distribution (D-05)."""
    # Create chunks with varying RMS — middle is normal, extreme high is loud, extreme low is quiet
    chunks = [
        _make_chunk(chunk_index=i, rms_mean=rms)
        for i, rms in enumerate([0.01, 0.05, 0.05, 0.05, 0.05, 0.5])
    ]
    levels = [compute_volume_level(c, chunks) for c in chunks]
    assert "quiet" in levels  # 0.01 is far below mean
    assert "loud" in levels   # 0.5 is far above mean
    assert "normal" in levels


def test_has_speech_flag():
    chunk_with = _make_chunk(transcript_text="hello")
    chunk_without = _make_chunk(transcript_text=None)
    assert bool(chunk_with.transcript) is True
    assert chunk_without.transcript is None


def test_has_ocr_flag():
    chunk_with = _make_chunk(ocr_text="STOP")
    chunk_without = _make_chunk(ocr_text=None)
    assert bool(chunk_with.ocr_results) is True
    assert chunk_without.ocr_results is None
