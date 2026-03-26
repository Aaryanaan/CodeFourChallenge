"""BM25 keyword index over transcripts and OCR text (IDX-03).

Indexes transcript + OCR text for keyword retrieval. Skips chunks with
neither transcript nor OCR (D-09). Persists as pickle (D-10).
Tokenization: lowercase + whitespace split (no stopword removal —
phrase queries like 'Miranda rights' depend on exact word presence).
"""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from videosearch.models import ChunkMetadata


class BM25Store:
    """BM25 keyword search over raw transcript text."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunk_docs: list[dict] = []
        self._initialized: bool = False  # True after build() or load() called

    @property
    def _corpus_size(self) -> int:
        return len(self._chunk_docs)

    @staticmethod
    def _get_transcript_text(chunk: ChunkMetadata) -> str:
        """Extract raw transcript text from chunk. Returns empty string if no transcript."""
        if not chunk.transcript:
            return ""
        return " ".join(seg.text for seg in chunk.transcript)

    @staticmethod
    def _get_ocr_text(chunk: ChunkMetadata) -> str:
        """Extract raw OCR text from chunk. Returns empty string if no OCR results."""
        if not chunk.ocr_results:
            return ""
        return " ".join(r.text for r in chunk.ocr_results)

    def build(self, chunks: list[ChunkMetadata]) -> None:
        """Build BM25 index from chunks. Skips chunks with no transcript and no OCR.

        Indexes both transcript and OCR text so keyword queries can match
        license plates, signs, and other visible text — not just spoken words.
        """
        corpus: list[list[str]] = []
        docs: list[dict] = []
        for chunk in chunks:
            transcript = self._get_transcript_text(chunk)
            ocr = self._get_ocr_text(chunk)
            text = " ".join(filter(None, [transcript, ocr]))
            if not text:
                continue  # skip chunks with no textual content
            corpus.append(text.lower().split())
            parts = []
            if transcript:
                parts.append(f"Transcript: {transcript}")
            if ocr:
                parts.append(f"OCR: {ocr}")
            docs.append(
                {
                    "video_id": chunk.video_id,
                    "chunk_index": chunk.chunk_index,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "duration": chunk.duration,
                    "combined_text": "\n".join(parts),
                }
            )
        self._initialized = True
        if not corpus:
            self._bm25 = None
            self._chunk_docs = []
            return
        self._bm25 = BM25Okapi(corpus)
        self._chunk_docs = docs

    def save(self, path: str | Path) -> None:
        """Persist BM25 index to pickle file (D-10)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_docs": self._chunk_docs}, f)

    def load(self, path: str | Path) -> None:
        """Load BM25 index from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_docs = data.get("chunk_docs", data.get("chunk_ids", []))
        self._initialized = True

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for query terms. Returns top-k results with scores > 0.

        Returns list of dicts with keys: video_id, chunk_index, score.
        Returns [] if corpus is empty (all-silent video — not an error).
        Raises RuntimeError if called before build() or load().
        """
        if not self._initialized:
            raise RuntimeError("BM25Store not built or loaded. Call build() or load() first.")
        if self._bm25 is None:
            return []  # built/loaded but corpus was empty
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {"score": float(scores[i]), **self._chunk_docs[i]}
            for i in top_indices
            if scores[i] > 0
        ]
