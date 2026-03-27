"""Embedding with Gemini API and local sentence-transformers fallback (IDX-01).

Primary: Gemini Text Embedding 004 via google-genai SDK.
Fallback: all-MiniLM-L6-v2 via sentence-transformers (local, no API needed).

The fallback activates automatically on Gemini quota exhaustion (429/RESOURCE_EXHAUSTED).
When fallback activates, the entire index must use the same embedder for consistency,
so it stays active for the lifetime of the GeminiEmbedder instance.
"""

import logging

from google import genai
from google.genai import types

from videosearch.config import Settings

logger = logging.getLogger(__name__)


class _LocalEmbedder:
    """Local embedding via sentence-transformers. Lazy-loaded."""

    def __init__(self, dimensions: int):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._dimensions = min(dimensions, 384)  # all-MiniLM-L6-v2 max is 384

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True).tolist()
        return vec[:self._dimensions]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True).tolist()
        return [v[:self._dimensions] for v in vecs]


class GeminiEmbedder:
    """Embeds text using Gemini embedding model. Implements Embedder protocol.

    Falls back to local sentence-transformers on Gemini quota exhaustion.
    """

    def __init__(self, settings: Settings):
        self._client = genai.Client(api_key=settings.google_api_key)
        self._model = settings.embedder_model  # "gemini-embedding-001"
        self._dimensions = settings.embedding_dimensions  # 768
        self._use_local = False
        self._local: _LocalEmbedder | None = None

    @property
    def dimensions(self) -> int:
        """Actual output dimension (may differ from config when using local fallback)."""
        if self._use_local:
            return min(self._dimensions, 384)
        return self._dimensions

    def _get_local(self) -> _LocalEmbedder:
        if self._local is None:
            logger.info("Initializing local embedding model (all-MiniLM-L6-v2)")
            self._local = _LocalEmbedder(self._dimensions)
        return self._local

    def _switch_to_local(self, exc: Exception) -> None:
        """Switch to local embedder on quota exhaustion."""
        if not self._use_local:
            logger.warning(
                "Gemini embedding quota exhausted — switching to local embeddings "
                "for this session: %s", exc
            )
            self._use_local = True

    def embed(self, text: str) -> list[float]:
        """Embed a single text for document indexing (RETRIEVAL_DOCUMENT)."""
        if self._use_local:
            return self._get_local().embed(text)
        try:
            result = self._client.models.embed_content(
                model=self._model,
                contents=[text],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self._dimensions,
                ),
            )
            return result.embeddings[0].values
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                self._switch_to_local(e)
                return self._get_local().embed(text)
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call. Max ~50 per call."""
        if self._use_local:
            return self._get_local().embed_batch(texts)
        try:
            result = self._client.models.embed_content(
                model=self._model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self._dimensions,
                ),
            )
            return [e.values for e in result.embeddings]
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                self._switch_to_local(e)
                return self._get_local().embed_batch(texts)
            raise

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (RETRIEVAL_QUERY). Phase 4 will use this."""
        if self._use_local:
            return self._get_local().embed(text)
        try:
            result = self._client.models.embed_content(
                model=self._model,
                contents=[text],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self._dimensions,
                ),
            )
            return result.embeddings[0].values
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                self._switch_to_local(e)
                return self._get_local().embed(text)
            raise
