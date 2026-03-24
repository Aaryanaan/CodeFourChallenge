"""Tests for GeminiEmbedder (IDX-01)."""
import pytest
from unittest.mock import MagicMock, patch

from videosearch.embedder import GeminiEmbedder
from videosearch.config import Settings


@pytest.fixture
def mock_genai_client():
    """Mock google.genai.Client for unit tests."""
    with patch("videosearch.embedder.genai.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        # Mock embed_content response
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_result
        yield mock_client


@pytest.fixture
def embedder(mock_genai_client):
    settings = Settings(google_api_key="test-key")
    return GeminiEmbedder(settings)


def test_embed_returns_list_of_floats(embedder):
    result = embedder.embed("test text")
    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


def test_embed_calls_api_with_correct_model(embedder, mock_genai_client):
    embedder.embed("test text")
    call_args = mock_genai_client.models.embed_content.call_args
    assert call_args.kwargs["model"] == "gemini-embedding-001"


def test_embed_uses_retrieval_document_task_type(embedder, mock_genai_client):
    embedder.embed("test text")
    call_args = mock_genai_client.models.embed_content.call_args
    config = call_args.kwargs["config"]
    assert config.task_type == "RETRIEVAL_DOCUMENT"


def test_embed_uses_configured_dimensions(embedder, mock_genai_client):
    embedder.embed("test text")
    call_args = mock_genai_client.models.embed_content.call_args
    config = call_args.kwargs["config"]
    assert config.output_dimensionality == 768


def test_embed_batch_returns_correct_count(mock_genai_client):
    # Setup batch mock
    mock_emb1 = MagicMock()
    mock_emb1.values = [0.1] * 768
    mock_emb2 = MagicMock()
    mock_emb2.values = [0.2] * 768
    mock_result = MagicMock()
    mock_result.embeddings = [mock_emb1, mock_emb2]
    mock_genai_client.models.embed_content.return_value = mock_result

    settings = Settings(google_api_key="test-key")
    embedder = GeminiEmbedder(settings)
    results = embedder.embed_batch(["text one", "text two"])
    assert len(results) == 2
    assert all(len(v) == 768 for v in results)


def test_embed_query_uses_retrieval_query_task_type(embedder, mock_genai_client):
    embedder.embed_query("search query")
    call_args = mock_genai_client.models.embed_content.call_args
    config = call_args.kwargs["config"]
    assert config.task_type == "RETRIEVAL_QUERY"


def test_embedder_satisfies_protocol():
    from videosearch.protocols import Embedder
    settings = Settings(google_api_key="test-key")
    with patch("videosearch.embedder.genai.Client"):
        e = GeminiEmbedder(settings)
        assert isinstance(e, Embedder)
