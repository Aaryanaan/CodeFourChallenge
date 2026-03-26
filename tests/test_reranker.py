"""Tests for ClaudeReranker.

Covers reranking with reasoning, disk caching, graceful degradation
on API failure, markdown fence stripping, and protocol conformance.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from videosearch.config import Settings
from videosearch.reranker import ClaudeReranker
from videosearch.protocols import Reranker


@pytest.fixture
def settings(tmp_path):
    """Create Settings with fake API key and temp cache dir."""
    return Settings(
        openrouter_api_key="fake-key",
        reranker_cache_dir=tmp_path / "reranker",
        _env_file=None,
    )


def _make_candidates(n: int = 3) -> list[dict]:
    """Build fake candidates with required fields."""
    return [
        {
            "video_id": f"video_{i}",
            "chunk_index": i,
            "combined_text": f"This is chunk {i} with some transcript text.",
            "start_time": float(i * 30),
            "end_time": float(i * 30 + 30),
            "score": 1.0 - (i * 0.1),
        }
        for i in range(n)
    ]


def _make_openrouter_response(rankings: list[dict]) -> MagicMock:
    """Create a mock httpx response mimicking OpenRouter format."""
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(rankings),
                }
            }
        ]
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestRerankReturnsReasoning:
    def test_rerank_returns_reasoning(self, settings):
        """Reranker returns candidates reordered with reasoning strings."""
        candidates = _make_candidates(3)

        # Reranker reorders: chunk 2 first, then 0, then 1
        rankings = [
            {"chunk_id": "video_2:2", "rank": 1, "reasoning": "Most relevant to query."},
            {"chunk_id": "video_0:0", "rank": 2, "reasoning": "Partially relevant."},
            {"chunk_id": "video_1:1", "rank": 3, "reasoning": "Least relevant."},
        ]

        with patch("videosearch.reranker.httpx.post") as mock_post:
            mock_post.return_value = _make_openrouter_response(rankings)
            reranker = ClaudeReranker(settings)
            result = reranker.rerank("test query", candidates, top_k=3)

        assert len(result) == 3
        assert result[0]["video_id"] == "video_2"
        assert result[0]["reasoning"] == "Most relevant to query."
        assert result[1]["video_id"] == "video_0"
        assert result[1]["reasoning"] == "Partially relevant."
        assert result[2]["video_id"] == "video_1"


class TestRerankCacheHit:
    def test_rerank_cache_hit(self, settings):
        """Second rerank with same query+candidates uses cache, no API call."""
        candidates = _make_candidates(3)
        rankings = [
            {"chunk_id": "video_0:0", "rank": 1, "reasoning": "Best match."},
            {"chunk_id": "video_1:1", "rank": 2, "reasoning": "OK match."},
            {"chunk_id": "video_2:2", "rank": 3, "reasoning": "Weak match."},
        ]

        with patch("videosearch.reranker.httpx.post") as mock_post:
            mock_post.return_value = _make_openrouter_response(rankings)
            reranker = ClaudeReranker(settings)

            result1 = reranker.rerank("test query", candidates, top_k=3)
            result2 = reranker.rerank("test query", candidates, top_k=3)

        assert result1[0]["video_id"] == result2[0]["video_id"]
        # httpx.post should only be called once
        assert mock_post.call_count == 1


class TestRerankDegradation:
    def test_rerank_degradation(self, settings):
        """On API failure, reranker returns original candidates[:top_k]."""
        candidates = _make_candidates(5)

        with patch("videosearch.reranker.httpx.post") as mock_post:
            mock_post.side_effect = Exception("API timeout")
            reranker = ClaudeReranker(settings)
            result = reranker.rerank("test query", candidates, top_k=3)

        assert len(result) == 3
        # Original order preserved
        assert result[0]["video_id"] == "video_0"
        assert result[1]["video_id"] == "video_1"
        assert result[2]["video_id"] == "video_2"


class TestRerankHttpErrorDegradation:
    def test_rerank_http_error_degrades_gracefully(self, settings):
        """On HTTP 402/429/500 error, reranker returns original candidates[:top_k]."""
        candidates = _make_candidates(5)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "402 Payment Required", request=MagicMock(), response=MagicMock(status_code=402)
        )

        with patch("videosearch.reranker.httpx.post", return_value=mock_response):
            reranker = ClaudeReranker(settings)
            result = reranker.rerank("test query", candidates, top_k=3)

        assert len(result) == 3
        assert result[0]["video_id"] == "video_0"


class TestRerankJsonParseStripsFences:
    def test_rerank_json_parse_strips_fences(self, settings):
        """Reranker handles markdown-fenced JSON responses."""
        candidates = _make_candidates(2)
        rankings = [
            {"chunk_id": "video_1:1", "rank": 1, "reasoning": "Top hit."},
            {"chunk_id": "video_0:0", "rank": 2, "reasoning": "Second."},
        ]

        # Wrap in markdown fences
        fenced_json = f"```json\n{json.dumps(rankings)}\n```"

        with patch("videosearch.reranker.httpx.post") as mock_post:
            resp = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"content": fenced_json}}]
            }
            resp.raise_for_status = MagicMock()
            mock_post.return_value = resp

            reranker = ClaudeReranker(settings)
            result = reranker.rerank("test query", candidates, top_k=2)

        assert len(result) == 2
        assert result[0]["video_id"] == "video_1"


class TestRerankCacheKeyIncludesCandidates:
    def test_rerank_cache_key_includes_candidates(self, settings):
        """Same query with different candidates produces different cache keys."""
        candidates_a = _make_candidates(2)
        candidates_b = [
            {
                "video_id": "other_0",
                "chunk_index": 0,
                "combined_text": "Different text.",
                "start_time": 0.0,
                "end_time": 30.0,
                "score": 0.9,
            },
            {
                "video_id": "other_1",
                "chunk_index": 1,
                "combined_text": "More different text.",
                "start_time": 30.0,
                "end_time": 60.0,
                "score": 0.8,
            },
        ]

        rankings_a = [
            {"chunk_id": "video_0:0", "rank": 1, "reasoning": "A."},
            {"chunk_id": "video_1:1", "rank": 2, "reasoning": "B."},
        ]
        rankings_b = [
            {"chunk_id": "other_0:0", "rank": 1, "reasoning": "X."},
            {"chunk_id": "other_1:1", "rank": 2, "reasoning": "Y."},
        ]

        with patch("videosearch.reranker.httpx.post") as mock_post:
            mock_post.side_effect = [
                _make_openrouter_response(rankings_a),
                _make_openrouter_response(rankings_b),
            ]
            reranker = ClaudeReranker(settings)

            result_a = reranker.rerank("same query", candidates_a, top_k=2)
            result_b = reranker.rerank("same query", candidates_b, top_k=2)

        # Both calls should hit the API (different cache keys)
        assert mock_post.call_count == 2
        assert result_a[0]["video_id"] == "video_0"
        assert result_b[0]["video_id"] == "other_0"


class TestRerankerProtocol:
    def test_reranker_protocol(self, settings):
        """ClaudeReranker satisfies Reranker protocol."""
        reranker = ClaudeReranker(settings)
        assert isinstance(reranker, Reranker)
