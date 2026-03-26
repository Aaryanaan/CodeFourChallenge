"""Hybrid retrieval engine combining vector, BM25, and metadata filter results (Phase 4).

Implements Retriever protocol. Fuses results via Reciprocal Rank Fusion (RRF)
with configurable per-source weights.
"""

from __future__ import annotations

from pathlib import Path

from videosearch.bm25_store import BM25Store
from videosearch.config import Settings
from videosearch.embedder import GeminiEmbedder
from videosearch.protocols import Classifier, Reranker
from videosearch.vector_store import LanceVectorStore


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[dict]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    Standard RRF formula: score(d) = sum_i weight_i / (k + rank_i(d))
    where rank is 1-indexed.

    Key normalization converts video_id to str and chunk_index to int so
    PyArrow int32 values (from LanceDB) deduplicate correctly with Python
    ints (from BM25Store).

    Args:
        ranked_lists: List of ranked result lists. Each item must have
                      "video_id" and "chunk_index" keys.
        weights:      Per-list weights (default: 1.0 for all lists).
        k:            RRF constant (default 60 per literature).

    Returns:
        Merged list of dicts sorted by rrf_score descending. Each dict
        preserves original keys from the first time the item was seen,
        with "rrf_score" added.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    # Accumulate scores and preserve first-seen item dict
    scores: dict[tuple, float] = {}
    representatives: dict[tuple, dict] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank_0indexed, item in enumerate(ranked_list):
            key = (str(item["video_id"]), int(item["chunk_index"]))
            rank_1indexed = rank_0indexed + 1
            scores[key] = scores.get(key, 0.0) + weight / (k + rank_1indexed)
            if key not in representatives:
                representatives[key] = dict(item)

    # Merge scores into result dicts
    result = []
    for key, score in scores.items():
        item = dict(representatives[key])
        item["rrf_score"] = score
        result.append(item)

    result.sort(key=lambda x: x["rrf_score"], reverse=True)
    return result


class HybridRetriever:
    """Combines vector, BM25, and metadata filter retrieval via RRF.

    Implements Retriever protocol: retrieve(query, top_k) -> list[dict].

    Three retrieval paths:
    1. Vector: embed_query -> LanceVectorStore.search
    2. BM25: BM25Store.search (keyword matching for transcripts)
    3. Filter: _detect_filters -> LanceVectorStore.search with filter_expr
               (only activated when query contains modality-specific terms)
    """

    def __init__(
        self,
        settings: Settings,
        classifier: Classifier | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = GeminiEmbedder(settings)
        self._vector_store = LanceVectorStore(index_dir=settings.index_dir, vector_dim=settings.embedding_dimensions)
        self._bm25_store = BM25Store()
        self._bm25_loaded = False

        bm25_path = settings.index_dir / "bm25.pkl"
        if bm25_path.exists():
            self._bm25_store.load(bm25_path)
            self._bm25_loaded = True

        self._classifier = classifier
        self._reranker = reranker
        self._default_weights = [
            settings.retrieval_vector_weight,
            settings.retrieval_bm25_weight,
            settings.retrieval_filter_weight,
        ]

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve top-k results via hybrid RRF fusion.

        When a classifier is present, uses its weights and query_type for
        filter mapping. When a reranker is present, widens the candidate pool
        to RERANK_POOL and reranks before returning top_k.

        Args:
            query:  Natural language query string.
            top_k:  Number of results to return.

        Returns:
            List of result dicts with original schema columns plus
            "rrf_score", sorted by rrf_score descending. When reranker is
            active, each dict also contains a "reasoning" key.
        """
        RERANK_POOL = 20  # wider pool for reranker input; output capped at top_k

        # Preflight: catch empty/missing index before spending an API call on embedding
        if self._vector_store.count() == 0:
            raise RuntimeError(
                "Vector index is empty — run `videosearch index` first."
            )

        # Step 1: Classify query (or fall back to heuristic)
        query_type = "mixed"
        if self._classifier:
            classification = self._classifier.classify(query)
            query_type = classification["query_type"]
            w = classification["weights"]
            weights = [w["vector"], w["bm25"], w["filter"]]
            filter_expr = self._map_filter(query, query_type)
        else:
            weights = list(self._default_weights)
            filter_expr = self._detect_filters(query)

        # Step 2: Retrieve from all paths (wider fetch when reranker present)
        fetch_k = max(top_k * 3, RERANK_POOL) if self._reranker else top_k * 2

        query_vector = self._embedder.embed_query(query)
        vector_results = self._vector_store.search(query_vector, top_k=fetch_k)

        # BM25 path (skipped gracefully if index was never built)
        bm25_results: list[dict] = []
        if self._bm25_loaded:
            bm25_results = self._bm25_store.search(query, top_k=fetch_k)

        # Filter path (conditional)
        filter_results: list[dict] = []
        if filter_expr is not None:
            filter_results = self._vector_store.search(
                query_vector, top_k=fetch_k, filter_expr=filter_expr
            )

        # Step 3: Build ranked lists and active weights for RRF
        lists: list[list[dict]] = [vector_results]
        active_weights: list[float] = [weights[0]]
        if bm25_results:
            lists.append(bm25_results)
            active_weights.append(weights[1])
        if filter_results:
            lists.append(filter_results)
            active_weights.append(weights[2])

        fused = reciprocal_rank_fusion(lists, active_weights)

        # Step 4: Rerank if available (input=top-20, output=top_k)
        if self._reranker:
            pool = fused[:RERANK_POOL]
            return self._reranker.rerank(query, pool, top_k, query_type=query_type)
        return fused[:top_k]

    def _map_filter(self, query: str, query_type: str) -> str | None:
        """Map classifier query_type + query keywords to a LanceDB filter expression.

        The classifier determines the TYPE; this method determines the specific
        FILTER expression based on type + query keywords.
        """
        q = query.lower()
        if query_type == "audio":
            if any(word in q for word in ["raise", "voice", "loud", "yell", "shout", "scream"]):
                return "has_raised_voice = true"
            if "quiet" in q or "silent" in q:
                return "volume_level = 'quiet'"
            # Default audio filter: raised voice (most common audio query)
            return "has_raised_voice = true"
        if query_type == "visual":
            if any(word in q for word in ["text", "sign", "license", "plate", "written"]):
                return "has_ocr = true"
        return None

    def _detect_filters(self, query: str) -> str | None:
        """Detect modality-specific filter expressions from query text.

        Legacy heuristic fallback — used only when no Classifier is injected.
        When a classifier is present, _map_filter() is called instead.

        Returns:
            A LanceDB filter expression string, or None if no filter applies.
        """
        q = query.lower()

        if any(word in q for word in ["raise", "voice", "loud", "yell", "shout", "scream"]):
            return "has_raised_voice = true"

        if any(word in q for word in ["text", "sign", "license", "plate", "written"]):
            return "has_ocr = true"

        if "quiet" in q or "silent" in q:
            return "volume_level = 'quiet'"

        return None
