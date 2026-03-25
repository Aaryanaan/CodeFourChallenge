"""Hybrid retrieval engine combining vector, BM25, and metadata filter results (Phase 4).

Implements Retriever protocol. Fuses results via Reciprocal Rank Fusion (RRF)
with configurable per-source weights.
"""

from pathlib import Path

from videosearch.bm25_store import BM25Store
from videosearch.config import Settings
from videosearch.embedder import GeminiEmbedder
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

    def __init__(self, settings: Settings) -> None:
        self._embedder = GeminiEmbedder(settings)
        self._vector_store = LanceVectorStore(index_dir=settings.index_dir)
        self._bm25_store = BM25Store()
        self._bm25_loaded = False

        bm25_path = settings.index_dir / "bm25.pkl"
        if bm25_path.exists():
            self._bm25_store.load(bm25_path)
            self._bm25_loaded = True

        self._weights = [
            settings.retrieval_vector_weight,
            settings.retrieval_bm25_weight,
            settings.retrieval_filter_weight,
        ]

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve top-k results via hybrid RRF fusion.

        Always runs vector and BM25 paths. Conditionally adds a metadata
        filter path when _detect_filters returns a non-None expression.

        Args:
            query:  Natural language query string.
            top_k:  Number of results to return.

        Returns:
            List of result dicts with original schema columns plus
            "rrf_score", sorted by rrf_score descending.
        """
        fetch_k = top_k * 2

        # Preflight: catch empty/missing index before spending an API call on embedding
        if self._vector_store.count() == 0:
            raise RuntimeError(
                "Vector index is empty — run `videosearch index` first."
            )

        # Vector path
        query_vector = self._embedder.embed_query(query)
        vector_results = self._vector_store.search(query_vector, top_k=fetch_k)

        # BM25 path (skipped gracefully if index was never built)
        bm25_results: list[dict] = []
        if self._bm25_loaded:
            bm25_results = self._bm25_store.search(query, top_k=fetch_k)

        # Filter path (conditional)
        filter_expr = self._detect_filters(query)
        filter_results: list[dict] = []
        if filter_expr is not None:
            filter_results = self._vector_store.search(
                query_vector, top_k=fetch_k, filter_expr=filter_expr
            )

        # Build ranked lists and weights for RRF
        lists: list[list[dict]] = [vector_results]
        active_weights: list[float] = [self._weights[0]]
        if bm25_results:
            lists.append(bm25_results)
            active_weights.append(self._weights[1])

        if filter_results:
            lists.append(filter_results)
            active_weights.append(self._weights[2])

        return reciprocal_rank_fusion(lists, active_weights)[:top_k]

    def _detect_filters(self, query: str) -> str | None:
        """Detect modality-specific filter expressions from query text.

        Heuristic keyword matching. Phase 6 will replace with LLM classifier.

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
