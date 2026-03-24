"""Tests for BM25Store (IDX-03)."""
import pytest

from videosearch.bm25_store import BM25Store
from videosearch.models import ChunkMetadata, TranscriptSegment


def _make_chunk(video_id, chunk_index, transcript_text=None):
    transcript = None
    if transcript_text:
        transcript = [TranscriptSegment(
            text=transcript_text, start=0.0, end=10.0,
            avg_logprob=-0.3,
            words=[{"word": w, "start": 0.0, "end": 1.0, "probability": 0.9}
                   for w in transcript_text.split()],
        )]
    return ChunkMetadata(
        video_id=video_id, chunk_index=chunk_index,
        start_time=0.0, end_time=30.0, duration=30.0,
        scene_type="detected", transcript=transcript,
    )


def test_build_skips_silent():
    store = BM25Store()
    chunks = [
        _make_chunk("v1", 0, "the officer read Miranda rights"),
        _make_chunk("v1", 1, None),  # silent — should be skipped
        _make_chunk("v1", 2, "then the suspect complied"),
    ]
    store.build(chunks)
    assert store._corpus_size == 2  # only 2 chunks with transcript


def test_search_phrase():
    store = BM25Store()
    chunks = [
        _make_chunk("v1", 0, "the officer read Miranda rights to the suspect"),
        _make_chunk("v1", 1, "the vehicle was parked on the street"),
        _make_chunk("v1", 2, "suspect was handcuffed and read Miranda rights"),
    ]
    store.build(chunks)
    results = store.search("Miranda rights", top_k=3)
    assert len(results) >= 2
    video_chunk_pairs = [(r["video_id"], r["chunk_index"]) for r in results]
    assert ("v1", 0) in video_chunk_pairs
    assert ("v1", 2) in video_chunk_pairs


def test_pickle_roundtrip(tmp_path):
    store = BM25Store()
    chunks = [
        _make_chunk("v1", 0, "the officer read Miranda rights"),
        _make_chunk("v1", 1, "the vehicle pulled over"),
    ]
    store.build(chunks)
    pkl_path = tmp_path / "bm25.pkl"
    store.save(str(pkl_path))

    store2 = BM25Store()
    store2.load(str(pkl_path))
    results = store2.search("Miranda rights", top_k=3)
    assert len(results) >= 1
    assert results[0]["video_id"] == "v1"
    assert results[0]["chunk_index"] == 0


def test_search_returns_scores():
    store = BM25Store()
    chunks = [_make_chunk("v1", 0, "hello world test text")]
    store.build(chunks)
    results = store.search("hello", top_k=5)
    assert len(results) >= 1
    assert "score" in results[0]
    assert results[0]["score"] > 0


def test_search_no_match_returns_empty():
    store = BM25Store()
    chunks = [_make_chunk("v1", 0, "hello world")]
    store.build(chunks)
    results = store.search("xylophone", top_k=5)
    assert len(results) == 0
