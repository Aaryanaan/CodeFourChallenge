"""ClaudeReranker: LLM reranking via Claude Sonnet on OpenRouter.

Implements the Reranker protocol. Reranks candidate search results using
Claude Sonnet via OpenRouter's OpenAI-compatible chat completions endpoint.
Populates per-result reasoning strings. Gracefully degrades to original
order on any API or parsing failure.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from videosearch.config import Settings

logger = logging.getLogger(__name__)

RERANKER_SYSTEM_PROMPT = """You are a precision reranker for a body-worn camera video search system.

Given a search query and candidate video chunks, rerank them by relevance.
Return ONLY a JSON array, no preamble or explanation outside the array.

Each element: {{"chunk_id": "video_id:chunk_index", "rank": N, "reasoning": "1-3 sentence explanation"}}

Rank 1 = most relevant. Only include candidates that are genuinely relevant.
If a candidate has no relevance to the query, exclude it from the output.

The query has been classified as type: {query_type}
Focus your relevance judgment accordingly:
- transcript: prioritize exact phrase matches in Transcript sections
- audio: prioritize audio feature indicators (raised voice, volume)
- visual: prioritize visual descriptions in Caption sections
- temporal: prioritize Lighting and time-of-day indicators in Caption sections
- mixed: consider all modalities equally
"""


class ClaudeReranker:
    """LLM reranker using Claude Sonnet via OpenRouter.

    Satisfies the Reranker protocol. Calls OpenRouter's chat completions
    endpoint via httpx. Caches results to disk keyed by query + candidate
    IDs. Gracefully degrades to original candidate order on failure.
    """

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.openrouter_api_key
        self._model = settings.reranker_model
        self._cache_dir = Path(settings.reranker_cache_dir)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
        *,
        query_type: str = "mixed",
    ) -> list[dict]:
        """Rerank candidates by relevance using Claude Sonnet.

        Args:
            query: The search query.
            candidates: List of candidate dicts with video_id, chunk_index,
                combined_text, start_time, end_time.
            top_k: Number of results to return.
            query_type: Classification type for prompt context (default "mixed").

        Returns:
            Reordered candidates[:top_k] with reasoning strings populated.
            On failure, returns original candidates[:top_k] without reasoning.
        """
        cache_key = self._cache_key(query, candidates)

        # Cache-first
        cached = self._load_cache(cache_key)
        if cached is not None:
            return self._apply_ranking(cached["rankings"], candidates, top_k)

        try:
            # Build prompts
            system_prompt = RERANKER_SYSTEM_PROMPT.format(query_type=query_type)
            user_prompt = self._build_rerank_prompt(query, query_type, candidates)

            # Call OpenRouter
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                },
                timeout=60.0,
            )

            response.raise_for_status()
            data = response.json()

            # Validate response structure — OpenRouter can return 200 with
            # an error body (e.g. {"error": {...}}) when auth fails or
            # upstream model is unavailable.
            if "error" in data:
                raise RuntimeError(
                    f"OpenRouter returned error: {data['error']}"
                )
            if "choices" not in data or not data["choices"]:
                raise RuntimeError(
                    f"OpenRouter response missing 'choices': {list(data.keys())}"
                )

            content = data["choices"][0]["message"]["content"]

            # Strip markdown fences if present
            content = self._strip_markdown_fences(content)

            rankings = json.loads(content)

            # Save to cache
            self._save_cache(cache_key, {
                "rankings": rankings,
                "model": self._model,
            })

            return self._apply_ranking(rankings, candidates, top_k)

        except Exception:
            logger.warning(
                "Reranker failed, returning original order",
                exc_info=True,
            )
            return candidates[:top_k]

    def _build_rerank_prompt(
        self, query: str, query_type: str, candidates: list[dict]
    ) -> str:
        """Build the user prompt with query and candidate texts."""
        parts = [f"Query: {query}\nQuery type: {query_type}\n\nCandidates:\n"]
        for i, c in enumerate(candidates):
            chunk_id = f"{c['video_id']}:{c['chunk_index']}"
            text = c.get("combined_text", "")[:1500]
            parts.append(f"[{i + 1}] chunk_id={chunk_id}\n{text}\n")
        parts.append("\nReturn JSON array of reranked results:")
        return "\n".join(parts)

    def _strip_markdown_fences(self, content: str) -> str:
        """Strip markdown code fences from LLM response."""
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            return match.group(1)
        return content

    def _apply_ranking(
        self, rankings: list[dict], candidates: list[dict], top_k: int
    ) -> list[dict]:
        """Reorder candidates by ranking and populate reasoning."""
        # Build lookup by chunk_id
        candidate_map = {
            f"{c['video_id']}:{c['chunk_index']}": c for c in candidates
        }

        result = []
        for entry in sorted(rankings, key=lambda x: x.get("rank", 999)):
            chunk_id = entry.get("chunk_id", "")
            if chunk_id in candidate_map:
                candidate = dict(candidate_map[chunk_id])
                candidate["reasoning"] = entry.get("reasoning", "")
                result.append(candidate)

        # If rankings didn't cover all candidates, append remaining
        seen = {r.get("chunk_id") for r in rankings}
        for c in candidates:
            cid = f"{c['video_id']}:{c['chunk_index']}"
            if cid not in seen:
                result.append(c)

        return result[:top_k]

    def _cache_key(self, query: str, candidates: list[dict]) -> str:
        """Compute cache key from query + sorted candidate IDs."""
        candidate_ids = ",".join(
            sorted(
                f"{c['video_id']}:{c['chunk_index']}" for c in candidates
            )
        )
        raw = query.lower().strip() + "|" + candidate_ids
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        """Return the cache file path for a given key."""
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> dict | None:
        """Load cached reranking if it exists."""
        path = self._cache_path(key)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        """Write reranking result to disk cache."""
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            **data,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(cache_data))
