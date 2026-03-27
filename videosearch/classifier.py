"""GeminiQueryClassifier: LLM-based query classification via OpenRouter.

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

import httpx

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
    """Query classifier using OpenRouter chat completions API.

    Satisfies the Classifier protocol. Uses disk cache keyed by
    SHA-256 hash of the normalized query string. The LLM determines
    query_type only; weight values are locked in WEIGHT_POLICY.

    Note: Class retains its original name for backwards compatibility with
    CLI and retriever imports, even though it now uses OpenRouter exclusively.
    """

    def __init__(self, settings: Settings) -> None:
        self._openrouter_key = settings.openrouter_api_key
        self._model = settings.classifier_model  # e.g. "google/gemini-2.0-flash"
        # Ensure model has provider prefix for OpenRouter
        if "/" not in self._model:
            self._model = f"google/{self._model}"
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

        query_type = self._classify_via_api(query)

        # Override LLM-returned weights with locked policy values
        if query_type not in WEIGHT_POLICY:
            query_type = "mixed"
        weights = WEIGHT_POLICY[query_type]

        result = {"query_type": query_type, "weights": weights}
        self._save_cache(key, result)
        return result

    def _classify_via_api(self, query: str) -> str:
        """Classify query via OpenRouter, falling back to heuristic on failure."""
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                        {"role": "user", "content": f'Classify this query: "{query}"'},
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise RuntimeError(f"OpenRouter error: {data['error']}")
            if "choices" not in data or not data["choices"]:
                raise RuntimeError(f"OpenRouter response missing choices: {list(data.keys())}")

            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed.get("query_type", "mixed")
        except Exception:
            logger.warning("Classifier failed, falling back to mixed weights", exc_info=True)
            return "mixed"

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
