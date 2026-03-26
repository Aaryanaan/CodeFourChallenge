"""Tests for the FastAPI server (videosearch/server.py)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


MOCK_RESULTS = [
    {
        "video_id": "bodycam_001",
        "chunk_index": 3,
        "start_time": 30.0,
        "end_time": 60.0,
        "combined_text": "Miranda rights read aloud to suspect",
        "visual_caption": "Officer standing beside patrol car",
        "transcript_snippet": "You have the right to remain silent",
        "rrf_score": 0.016,
        "reasoning": "Contains Miranda rights language",
    },
]


@patch("videosearch.server.ClaudeReranker")
@patch("videosearch.server.GeminiQueryClassifier")
@patch("videosearch.server.HybridRetriever")
@patch("videosearch.server.Settings")
def test_search_endpoint(mock_settings, mock_retriever_cls, mock_classifier_cls, mock_reranker_cls):
    """POST /search returns JSON with query and results."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = MOCK_RESULTS
    mock_retriever_cls.return_value = mock_retriever

    from videosearch.server import app
    client = TestClient(app)

    response = client.post("/search", json={"query": "Miranda rights", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Miranda rights"
    assert len(data["results"]) == 1
    r = data["results"][0]
    assert r["video_id"] == "bodycam_001"
    assert r["start_time"] == 30.0
    assert r["end_time"] == 60.0
    assert r["score"] == 0.016
    assert "Miranda" in r["transcript_snippet"] or "Miranda" in r["visual_caption"] or len(r["transcript_snippet"]) >= 0
    assert "reasoning" in r

    mock_retriever.retrieve.assert_called_once_with("Miranda rights", top_k=5)


@patch("videosearch.server.ClaudeReranker")
@patch("videosearch.server.GeminiQueryClassifier")
@patch("videosearch.server.HybridRetriever")
@patch("videosearch.server.Settings")
def test_search_endpoint_empty_results(mock_settings, mock_retriever_cls, mock_classifier_cls, mock_reranker_cls):
    """POST /search with no matching results returns empty list."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever_cls.return_value = mock_retriever

    from videosearch.server import app
    client = TestClient(app)

    response = client.post("/search", json={"query": "nonexistent"})
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []


@patch("videosearch.server.ClaudeReranker")
@patch("videosearch.server.GeminiQueryClassifier")
@patch("videosearch.server.HybridRetriever")
@patch("videosearch.server.Settings")
def test_search_endpoint_validation_error(mock_settings, mock_retriever_cls, mock_classifier_cls, mock_reranker_cls):
    """POST /search without query field returns 422."""
    from videosearch.server import app
    client = TestClient(app)

    response = client.post("/search", json={"top_k": 5})
    assert response.status_code == 422


@patch("videosearch.server.ClaudeReranker")
@patch("videosearch.server.GeminiQueryClassifier")
@patch("videosearch.server.HybridRetriever")
@patch("videosearch.server.Settings")
def test_search_endpoint_default_top_k(mock_settings, mock_retriever_cls, mock_classifier_cls, mock_reranker_cls):
    """POST /search without top_k uses default of 10."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever_cls.return_value = mock_retriever

    from videosearch.server import app
    client = TestClient(app)

    response = client.post("/search", json={"query": "test"})
    assert response.status_code == 200
    mock_retriever.retrieve.assert_called_once_with("test", top_k=10)
