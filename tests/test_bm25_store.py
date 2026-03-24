"""Tests for BM25Store keyword index over raw transcripts (IDX-03)."""

import tempfile
from pathlib import Path

import pytest

from videosearch.bm25_store import BM25Store
from videosearch.models import ChunkMetadata, TranscriptSegment


def _make_segment(text: str) -> TranscriptSegment:
    """Helper to create a TranscriptSegment with minimal fields."""
    return TranscriptSegment(
        text=text,
        start=0.0,
        end=1.0,
        avg_logprob=-0.3,
        words=[],
    )


def _make_chunk(
    video_id: str,
    chunk_index: int,
    transcript_texts: list[str] | None = None,
) -> ChunkMetadata:
    """Helper to create a ChunkMetadata with optional transcript segments."""
    transcript = None
    if transcript_texts is not None:
        transcript = [_make_segment(t) for t in transcript_texts]
    return ChunkMetadata(
        video_id=video_id,
        chunk_index=chunk_index,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        scene_type="detected",
        transcript=transcript,
    )


class TestBM25Store:
    """BM25Store test suite."""

    @pytest.fixture()
    def sample_chunks(self):
        """Corpus with 4 chunks: 0 and 2 mention Miranda rights, 1 silent, 3 unrelated."""
        return [
            _make_chunk("v1", 0, ["The officer read Miranda rights to the suspect"]),
            _make_chunk("v1", 1, None),  # silent — no transcript
            _make_chunk("v1", 2, ["Miranda rights were read again at the scene"]),
            _make_chunk("v1", 3, ["The vehicle was stopped at the intersection"]),
        ]

    def test_build_skips_silent(self, sample_chunks):
        """build() with 4 chunks (3 have transcript, 1 silent) yields _corpus_size == 3."""
        store = BM25Store()
        store.build(sample_chunks)
        assert store._corpus_size == 3

    def test_search_phrase(self, sample_chunks):
        """search('Miranda rights') returns chunks 0 and 2 which contain that phrase."""
        store = BM25Store()
        store.build(sample_chunks)
        results = store.search("Miranda rights")
        result_indices = {r["chunk_index"] for r in results}
        assert result_indices == {0, 2}

    def test_pickle_roundtrip(self, sample_chunks, tmp_path):
        """save then load produces identical search results."""
        store = BM25Store()
        store.build(sample_chunks)
        original_results = store.search("Miranda rights")

        pkl_path = tmp_path / "bm25.pkl"
        store.save(pkl_path)

        loaded_store = BM25Store()
        loaded_store.load(pkl_path)
        loaded_results = loaded_store.search("Miranda rights")

        assert loaded_results == original_results

    def test_search_returns_scores(self, sample_chunks):
        """Results have 'score' key with value > 0."""
        store = BM25Store()
        store.build(sample_chunks)
        results = store.search("Miranda rights")
        assert len(results) > 0
        for r in results:
            assert "score" in r
            assert r["score"] > 0

    def test_search_no_match_returns_empty(self, sample_chunks):
        """Search for non-existent word returns []."""
        store = BM25Store()
        store.build(sample_chunks)
        results = store.search("xylophone")
        assert results == []
