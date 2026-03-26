"""GeminiQueryClassifier: LLM-based query classification via Gemini Flash.

Implements the Classifier protocol. Classifies search queries into one of
five types (visual, audio, transcript, temporal, mixed) and returns locked
weight policies for hybrid retrieval. Uses disk cache to avoid redundant
API calls.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

from videosearch.config import Settings

logger = logging.getLogger(__name__)

CLASSIFIER_SYSTEM_PROMPT = """You classify search queries for a body-worn camera video search system.

Given a query, return JSON with:
- query_type: one of "visual", "audio", "transcript", "temporal", "mixed"
- weights: {"vector": float, "bm25": float, "filter": float}

Query type definitions:
- "transcript": Query about specific spoken words or phrases. BM25 keyword search is most effective.
- "audio": Query about sounds, voice volume, or prosody. Metadata filters (raised voice, volume level) are most effective.
- "visual": Query about appearance, objects, clothing, actions, or scenes. Vector similarity search is most effective.
- "temporal": Query about time of day, lighting conditions, or temporal context. Vector search over captions (which include Lighting: field) is most effective.
- "mixed": Query combining multiple modalities. Use balanced weights.

Weight guidelines (higher = more influence):
- Dominant path: 2.5
- Secondary path: 0.3-0.5
- Disabled path: 0.0

Examples:
Query: "Find all interactions where an officer reads Miranda rights"
{"query_type": "transcript", "weights": {"vector": 0.3, "bm25": 2.5, "filter": 0.0}}

Query: "Find every moment where someone raises their voice"
{"query_type": "audio", "weights": {"vector": 0.3, "bm25": 0.5, "filter": 2.5}}

Query: "Locate all footage containing a person in a red shirt"
{"query_type": "visual", "weights": {"vector": 2.5, "bm25": 0.3, "filter": 0.0}}

Query: "Find all instances of a vehicle being pulled over at night"
{"query_type": "temporal", "weights": {"vector": 2.0, "bm25": 0.3, "filter": 0.0}}

Query: "Find moments where a suspect is being handcuffed"
{"query_type": "visual", "weights": {"vector": 2.5, "bm25": 0.3, "filter": 0.0}}

Query: "Find all license plates visible in the footage"
{"query_type": "visual", "weights": {"vector": 2.5, "bm25": 0.3, "filter": 0.0}}
"""

# Locked weight policies per query type. The LLM only determines query_type;
# weights are deterministic code, not LLM output.
WEIGHT_POLICY: dict[str, dict[str, float]] = {
    "transcript": {"vector": 0.3, "bm25": 2.5, "filter": 0.0},
    "audio": {"vector": 0.3, "bm25": 0.5, "filter": 2.5},
    "visual": {"vector": 2.5, "bm25": 0.3, "filter": 0.0},
    "temporal": {"vector": 2.0, "bm25": 0.3, "filter": 0.0},
    "mixed": {"vector": 1.0, "bm25": 1.0, "filter": 0.5},
}


class GeminiQueryClassifier:
    """Query classifier using Gemini Flash via Google GenAI SDK.

    Satisfies the Classifier protocol. Uses disk cache keyed by
    SHA-256 hash of the normalized query string. The LLM determines
    query_type only; weight values are locked in WEIGHT_POLICY.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(api_key=settings.google_api_key)
        self._model = settings.classifier_model
        self._cache_dir = Path(settings.classifier_cache_dir)

    def classify(self, query: str) -> dict:
        """Classify a search query into a type with retrieval weights.

        Args:
            query: Natural language search query.

        Returns:
            Dict with keys: "query_type" (str), "weights" (dict with
            "vector", "bm25", "filter" float values).
        """
        key = self._cache_key(query)

        # Cache-first: return cached value without any API call
        cached = self._load_cache(key)
        if cached is not None:
            return {"query_type": cached["query_type"], "weights": cached["weights"]}

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[f'Classify this query: "{query}"'],
                config=types.GenerateContentConfig(
                    system_instruction=CLASSIFIER_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            parsed = json.loads(response.text)
            query_type = parsed.get("query_type", "mixed")
        except Exception:
            logger.warning(
                "Classifier failed, falling back to mixed weights",
                exc_info=True,
            )
            query_type = "mixed"

        # Override LLM-returned weights with locked policy values
        if query_type not in WEIGHT_POLICY:
            query_type = "mixed"
        weights = WEIGHT_POLICY[query_type]

        result = {"query_type": query_type, "weights": weights}
        self._save_cache(key, result)
        return result

    def _cache_key(self, query: str) -> str:
        """Compute cache key from normalized query."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        """Return the cache file path for a given key."""
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> dict | None:
        """Load cached classification if it exists."""
        path = self._cache_path(key)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        """Write classification result to disk cache."""
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            **data,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(cache_data))
