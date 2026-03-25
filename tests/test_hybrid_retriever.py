"""Tests for hybrid retrieval engine (Phase 4, Plan 1)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from videosearch.models import SearchResult
from videosearch.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion tests (pure function, no mocking needed)
# ---------------------------------------------------------------------------


def test_rrf_fusion():
    """Items appearing in both lists rank first; correct merged order."""
    list1 = [
        {"video_id": "v1", "chunk_index": 0},
        {"video_id": "v1", "chunk_index": 1},
    ]
    list2 = [
        {"video_id": "v1", "chunk_index": 1},
        {"video_id": "v1", "chunk_index": 2},
    ]
    result = reciprocal_rank_fusion([list1, list2])
    scores = {r["chunk_index"]: r["rrf_score"] for r in result}
    # chunk_index=1 appears in both lists -> highest score
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_rrf_weights():
    """Higher weight for list1 boosts its items."""
    # chunk_index=0 only in list1; chunk_index=1 only in list2
    list1 = [{"video_id": "v1", "chunk_index": 0}]
    list2 = [{"video_id": "v1", "chunk_index": 1}]
    result = reciprocal_rank_fusion([list1, list2], weights=[10.0, 1.0])
    scores = {r["chunk_index"]: r["rrf_score"] for r in result}
    # list1 weight is 10x bigger -> chunk_index=0 should outrank chunk_index=1
    assert scores[0] > scores[1]


def test_rrf_empty_list():
    """Empty list among inputs doesn't crash."""
    list1 = [{"video_id": "v1", "chunk_index": 0}]
    list2: list = []
    result = reciprocal_rank_fusion([list1, list2])
    assert len(result) == 1
    assert result[0]["chunk_index"] == 0


def test_rrf_key_normalization():
    """PyArrow int32 and Python int keys deduplicate correctly."""
    list1 = [{"video_id": "v1", "chunk_index": np.int32(5)}]
    list2 = [{"video_id": "v1", "chunk_index": int(5)}]
    result = reciprocal_rank_fusion([list1, list2])
    # Should deduplicate: only 1 result
    assert len(result) == 1
    # Score should reflect appearance in both lists
    assert result[0]["rrf_score"] > 0


# ---------------------------------------------------------------------------
# SearchResult model test
# ---------------------------------------------------------------------------


def test_search_result_model():
    """SearchResult validates correctly with defaults."""
    sr = SearchResult(
        video_id="v1",
        chunk_index=0,
        start_time=0.0,
        end_time=10.0,
        score=0.5,
    )
    assert sr.video_id == "v1"
    assert sr.chunk_index == 0
    assert sr.start_time == 0.0
    assert sr.end_time == 10.0
    assert sr.score == 0.5
    assert sr.reasoning == ""
    assert sr.transcript_snippet == ""


# ---------------------------------------------------------------------------
# HybridRetriever tests (mock all external components)
# ---------------------------------------------------------------------------


def _make_retriever():
    """Create a HybridRetriever with all dependencies mocked."""
    with (
        patch("videosearch.hybrid_retriever.GeminiEmbedder") as MockEmb,
        patch("videosearch.hybrid_retriever.LanceVectorStore") as MockVS,
        patch("videosearch.hybrid_retriever.BM25Store") as MockBM25,
    ):
        from videosearch.config import Settings
        settings = Settings(
            google_api_key="fake",
            retrieval_vector_weight=1.0,
            retrieval_bm25_weight=1.0,
            retrieval_filter_weight=0.5,
        )
        retriever = HybridRetriever(settings)

    return retriever, MockEmb, MockVS, MockBM25


def _build_retriever_with_mocks():
    """Return a retriever with accessible mock instances on _embedder, _vector_store, _bm25_store."""
    from videosearch.config import Settings

    settings = Settings(
        google_api_key="fake",
        retrieval_vector_weight=1.0,
        retrieval_bm25_weight=1.0,
        retrieval_filter_weight=0.5,
    )

    with (
        patch("videosearch.hybrid_retriever.GeminiEmbedder") as MockEmb,
        patch("videosearch.hybrid_retriever.LanceVectorStore") as MockVS,
        patch("videosearch.hybrid_retriever.BM25Store") as MockBM25,
    ):
        mock_emb_instance = MagicMock()
        mock_vs_instance = MagicMock()
        mock_bm25_instance = MagicMock()

        MockEmb.return_value = mock_emb_instance
        MockVS.return_value = mock_vs_instance
        MockBM25.return_value = mock_bm25_instance

        retriever = HybridRetriever(settings)
        retriever._bm25_loaded = True  # simulate bm25.pkl present and loaded

    return retriever, mock_emb_instance, mock_vs_instance, mock_bm25_instance


def test_hybrid_three_paths():
    """retrieve() calls embed_query, vector_store.search, bm25_store.search."""
    retriever, mock_emb, mock_vs, mock_bm25 = _build_retriever_with_mocks()

    mock_emb.embed_query.return_value = [0.1] * 768
    mock_vs.search.return_value = [
        {"video_id": "v1", "chunk_index": 0, "start_time": 0.0, "end_time": 10.0, "combined_text": "hello", "_distance": 0.1},
    ]
    mock_bm25.search.return_value = [
        {"video_id": "v1", "chunk_index": 1, "score": 2.5},
    ]

    results = retriever.retrieve("find something", top_k=5)

    mock_emb.embed_query.assert_called_once_with("find something")
    mock_vs.search.assert_called()
    mock_bm25.search.assert_called_once_with("find something", top_k=10)

    assert isinstance(results, list)


def test_result_fields():
    """Results from retrieve() contain required keys."""
    retriever, mock_emb, mock_vs, mock_bm25 = _build_retriever_with_mocks()

    mock_emb.embed_query.return_value = [0.1] * 768
    mock_vs.search.return_value = [
        {
            "video_id": "v1", "chunk_index": 0,
            "start_time": 0.0, "end_time": 10.0,
            "combined_text": "officer reads rights",
            "_distance": 0.1,
        }
    ]
    mock_bm25.search.return_value = [
        {"video_id": "v1", "chunk_index": 0, "score": 3.0},
    ]

    results = retriever.retrieve("Miranda rights", top_k=3)

    assert len(results) > 0
    first = results[0]
    for key in ["video_id", "chunk_index", "start_time", "end_time", "combined_text", "rrf_score"]:
        assert key in first, f"Missing key: {key}"


def test_raised_voice_filter():
    """Query containing voice/raise keywords triggers has_raised_voice filter."""
    retriever, mock_emb, mock_vs, mock_bm25 = _build_retriever_with_mocks()

    mock_emb.embed_query.return_value = [0.0] * 768
    mock_vs.search.return_value = []
    mock_bm25.search.return_value = []

    retriever.retrieve("someone raises their voice", top_k=5)

    # vector_store.search should be called at least twice:
    # once normal + once with filter_expr for raised_voice
    calls = mock_vs.search.call_args_list
    filter_calls = [c for c in calls if c.kwargs.get("filter_expr") == "has_raised_voice = true"
                    or (len(c.args) >= 3 and c.args[2] == "has_raised_voice = true")]
    # Check any call has filter_expr with raised_voice
    has_filter = any(
        "has_raised_voice" in str(c) for c in calls
    )
    assert has_filter, f"Expected raised_voice filter in calls: {calls}"


def test_miranda_bm25():
    """BM25 results appear in fused output for transcript-heavy query."""
    retriever, mock_emb, mock_vs, mock_bm25 = _build_retriever_with_mocks()

    mock_emb.embed_query.return_value = [0.0] * 768
    mock_vs.search.return_value = []
    mock_bm25.search.return_value = [
        {"video_id": "v1", "chunk_index": 5, "score": 10.0},
        {"video_id": "v1", "chunk_index": 6, "score": 8.0},
    ]

    results = retriever.retrieve("officer reads Miranda rights", top_k=5)

    mock_bm25.search.assert_called()
    # BM25 results should be in final output since vector returns nothing
    result_chunk_indices = [r["chunk_index"] for r in results]
    assert 5 in result_chunk_indices or 6 in result_chunk_indices


def test_no_filter_for_generic_query():
    """Generic query does NOT trigger any metadata filter."""
    retriever, mock_emb, mock_vs, mock_bm25 = _build_retriever_with_mocks()

    mock_emb.embed_query.return_value = [0.0] * 768
    mock_vs.search.return_value = []
    mock_bm25.search.return_value = []

    retriever.retrieve("what happened yesterday", top_k=5)

    # vector_store.search called only once (no filter call)
    calls = mock_vs.search.call_args_list
    assert len(calls) == 1, f"Expected 1 vector search call, got {len(calls)}: {calls}"
    # The single call should have no filter_expr
    call_kwargs = calls[0].kwargs
    call_args = calls[0].args
    filter_val = call_kwargs.get("filter_expr", None)
    assert filter_val is None
