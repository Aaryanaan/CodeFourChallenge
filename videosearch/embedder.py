"""Gemini embedding via google-genai SDK (IDX-01)."""

from google import genai
from google.genai import types

from videosearch.config import Settings


class GeminiEmbedder:
    """Embeds text using Gemini embedding model. Implements Embedder protocol."""

    def __init__(self, settings: Settings):
        self._client = genai.Client(api_key=settings.google_api_key)
        self._model = settings.embedder_model  # "gemini-embedding-001"
        self._dimensions = settings.embedding_dimensions  # 768

    def embed(self, text: str) -> list[float]:
        """Embed a single text for document indexing (RETRIEVAL_DOCUMENT)."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self._dimensions,
            ),
        )
        return result.embeddings[0].values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call. Max ~50 per call."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self._dimensions,
            ),
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (RETRIEVAL_QUERY). Phase 4 will use this."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self._dimensions,
            ),
        )
        return result.embeddings[0].values
