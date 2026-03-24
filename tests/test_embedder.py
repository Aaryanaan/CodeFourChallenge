"""Tests for GeminiEmbedder (IDX-01).

All tests mock the google-genai client to avoid real API calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from videosearch.config import Settings
from videosearch.embedder import GeminiEmbedder
from videosearch.protocols import Embedder


@pytest.fixture
def mock_settings():
    """Settings with test defaults."""
    return Settings(
        google_api_key="test-api-key",
        embedder_model="gemini-embedding-001",
        embedding_dimensions=768,
    )


@pytest.fixture
def fake_embedding():
    """A fake 768-dim embedding."""
    return [0.1] * 768


@pytest.fixture
def mock_client(fake_embedding):
    """Mock genai.Client with embed_content returning fake embeddings."""
    client = MagicMock()
    embedding_obj = MagicMock()
    embedding_obj.values = fake_embedding
    result = MagicMock()
    result.embeddings = [embedding_obj]
    client.models.embed_content.return_value = result
    return client


@pytest.fixture
def embedder(mock_settings, mock_client):
    """GeminiEmbedder with mocked client."""
    with patch("videosearch.embedder.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        emb = GeminiEmbedder(mock_settings)
    emb._client = mock_client
    return emb


def test_embed_returns_list_of_floats(embedder, fake_embedding):
    """embed() returns list[float] with len==768."""
    result = embedder.embed("test text")
    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


def test_embed_calls_api_with_correct_model(embedder, mock_client):
    """API called with model='gemini-embedding-001'."""
    embedder.embed("test text")
    call_kwargs = mock_client.models.embed_content.call_args
    assert call_kwargs.kwargs["model"] == "gemini-embedding-001"


def test_embed_uses_retrieval_document_task_type(embedder, mock_client):
    """embed() uses RETRIEVAL_DOCUMENT task type."""
    embedder.embed("test text")
    call_kwargs = mock_client.models.embed_content.call_args
    config = call_kwargs.kwargs["config"]
    assert config.task_type == "RETRIEVAL_DOCUMENT"


def test_embed_uses_configured_dimensions(embedder, mock_client):
    """embed() passes output_dimensionality=768."""
    embedder.embed("test text")
    call_kwargs = mock_client.models.embed_content.call_args
    config = call_kwargs.kwargs["config"]
    assert config.output_dimensionality == 768


def test_embed_batch_returns_correct_count(mock_settings, fake_embedding):
    """embed_batch(["a","b"]) returns 2 embeddings."""
    mock_client = MagicMock()
    emb_a = MagicMock()
    emb_a.values = fake_embedding
    emb_b = MagicMock()
    emb_b.values = fake_embedding
    result = MagicMock()
    result.embeddings = [emb_a, emb_b]
    mock_client.models.embed_content.return_value = result

    with patch("videosearch.embedder.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        embedder = GeminiEmbedder(mock_settings)
    embedder._client = mock_client

    results = embedder.embed_batch(["text a", "text b"])
    assert len(results) == 2
    assert all(len(e) == 768 for e in results)


def test_embed_query_uses_retrieval_query_task_type(embedder, mock_client):
    """embed_query() uses RETRIEVAL_QUERY task type."""
    embedder.embed_query("search query")
    call_kwargs = mock_client.models.embed_content.call_args
    config = call_kwargs.kwargs["config"]
    assert config.task_type == "RETRIEVAL_QUERY"


def test_embedder_satisfies_protocol(embedder):
    """GeminiEmbedder satisfies the Embedder protocol."""
    assert isinstance(embedder, Embedder)
