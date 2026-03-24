"""BM25 keyword index over raw transcripts (IDX-03).

Indexes transcript text only (D-08), skips silent chunks (D-09),
persists as pickle (D-10). Tokenization: lowercase + whitespace split
(no stopword removal — phrase queries like 'Miranda rights' depend on
exact word presence).
"""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from videosearch.models import ChunkMetadata


class BM25Store:
    """BM25 keyword search over raw transcript text."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[dict] = []  # [{"video_id": ..., "chunk_index": ...}]

    @property
    def _corpus_size(self) -> int:
        return len(self._chunk_ids)

    @staticmethod
    def _get_transcript_text(chunk: ChunkMetadata) -> str:
        """Extract raw transcript text from chunk. Returns empty string if no transcript."""
        if not chunk.transcript:
            return ""
        return " ".join(seg.text for seg in chunk.transcript)

    def build(self, chunks: list[ChunkMetadata]) -> None:
        """Build BM25 index from chunks. Skips chunks with no transcript (D-09)."""
        corpus: list[list[str]] = []
        ids: list[dict] = []
        for chunk in chunks:
            text = self._get_transcript_text(chunk)
            if not text:
                continue  # D-09: skip silent chunks
            corpus.append(text.lower().split())
            ids.append({"video_id": chunk.video_id, "chunk_index": chunk.chunk_index})
        self._bm25 = BM25Okapi(corpus)
        self._chunk_ids = ids

    def save(self, path: str | Path) -> None:
        """Persist BM25 index to pickle file (D-10)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self._chunk_ids}, f)

    def load(self, path: str | Path) -> None:
        """Load BM25 index from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for query terms. Returns top-k results with scores > 0.

        Returns list of dicts with keys: video_id, chunk_index, score.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Store not built or loaded. Call build() or load() first.")
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {"score": float(scores[i]), **self._chunk_ids[i]}
            for i in top_indices
            if scores[i] > 0
        ]
