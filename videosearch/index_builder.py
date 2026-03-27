"""Orchestrates index building across all stores (IDX-01, IDX-02, IDX-03, IDX-04).

Reads ChunkMetadata, computes derived fields, generates embeddings,
and populates LanceDB vector store + BM25 keyword index.
"""

import logging
from pathlib import Path

from videosearch.bm25_store import BM25Store
from videosearch.config import Settings
from videosearch.embedder import GeminiEmbedder
from videosearch.metadata_writer import MetadataWriter
from videosearch.models import ChunkMetadata
from videosearch.vector_store import LanceVectorStore

logger = logging.getLogger(__name__)


def build_combined_text(chunk: ChunkMetadata) -> str:
    """Build labeled combined text from transcript + OCR + caption (D-01, D-23).

    Format: 'Transcript: {text}\\nOCR: {text}\\nCaption: {text}'
    Returns empty string if all are missing (D-03 -- skip for embedding).
    Does NOT include audio feature descriptors (D-02).
    Caption goes last (per D-23 -- most token-dense, won't crowd out exact matches).
    """
    parts = []
    if chunk.transcript:
        transcript_text = " ".join(seg.text for seg in chunk.transcript)
        if transcript_text.strip():
            parts.append(f"Transcript: {transcript_text}")
    if chunk.ocr_results:
        ocr_text = " ".join(r.text for r in chunk.ocr_results)
        if ocr_text.strip():
            parts.append(f"OCR: {ocr_text}")
    if chunk.visual_caption:
        parts.append(f"Caption: {chunk.visual_caption}")
    return "\n".join(parts)


def compute_volume_level(
    chunk: ChunkMetadata,
    all_chunks: list[ChunkMetadata],
    stddev_threshold: float = 2.0,
) -> str:
    """Compute volume bin relative to per-video RMS distribution (D-05).

    Returns 'quiet', 'normal', or 'loud' based on how many standard
    deviations the chunk's RMS is from the video mean.
    """
    video_chunks = [
        c for c in all_chunks
        if c.video_id == chunk.video_id and c.audio_features is not None
    ]
    if not video_chunks or chunk.audio_features is None:
        return "normal"

    rms_values = [c.audio_features.rms_mean for c in video_chunks]
    n = len(rms_values)
    mean = sum(rms_values) / n
    variance = sum((x - mean) ** 2 for x in rms_values) / n
    stddev = variance ** 0.5

    if stddev == 0:
        return "normal"

    rms = chunk.audio_features.rms_mean
    if rms > mean + stddev_threshold * stddev:
        return "loud"
    elif rms < mean - stddev_threshold * stddev:
        return "quiet"
    return "normal"


class IndexBuilder:
    """Orchestrates building vector + BM25 indices from chunk metadata.

    Usage:
        builder = IndexBuilder(settings)
        builder.build_index(["video_001", "video_002"])
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()
        self._embedder = GeminiEmbedder(self._settings)
        self._vector_store: LanceVectorStore | None = None
        self._bm25_store = BM25Store()
        self._metadata_writer = MetadataWriter(
            metadata_dir=self._settings.metadata_dir
        )

    def _get_vector_store(self, vector_dim: int | None = None) -> LanceVectorStore:
        """Get or create the vector store with the correct dimensions."""
        dim = vector_dim or self._settings.embedding_dimensions
        if self._vector_store is None or self._vector_store._vector_dim != dim:
            self._vector_store = LanceVectorStore(
                index_dir=self._settings.index_dir,
                vector_dim=dim,
            )
        return self._vector_store

    def _is_video_indexed(self, video_id: str, expected_chunks: int) -> bool:
        """Check if video already has expected number of rows in vector store (D-05)."""
        existing = self._get_vector_store().count_by_video(video_id)
        return existing == expected_chunks

    def build_index(self, video_ids: list[str], force: bool = False) -> dict:
        """Build full index for given videos. Returns stats dict.

        1. Load all chunks from metadata files (skip already-indexed per IDX-05)
        2. Compute derived fields (combined_text, volume_level, flags)
        3. Embed non-empty chunks via GeminiEmbedder
        4. Upsert into LanceDB vector store
        5. Build + save BM25 index from transcripts
        """
        # Step 1: Load all chunks, with incremental skip detection
        all_chunks: list[ChunkMetadata] = []
        skipped_videos: list[str] = []
        for vid in video_ids:
            chunks = self._metadata_writer.load(vid)
            if not force and self._is_video_indexed(vid, len(chunks)):
                logger.info("Skipping %s (already indexed, %d chunks)", vid, len(chunks))
                skipped_videos.append(vid)
                continue
            all_chunks.extend(chunks)
            logger.info("Loaded %d chunks from %s", len(chunks), vid)

        # Step 2: Prepare rows for vector store
        rows_to_embed: list[tuple[ChunkMetadata, str]] = []
        for chunk in all_chunks:
            combined = build_combined_text(chunk)
            if combined:  # D-03: skip empty chunks for embedding
                rows_to_embed.append((chunk, combined))

        logger.info(
            "Embedding %d/%d chunks (skipped %d empty)",
            len(rows_to_embed), len(all_chunks),
            len(all_chunks) - len(rows_to_embed),
        )

        # Step 3: Batch embed
        # Check existing table dimension to detect incompatible embedder switch
        embedding_failed = False
        if not force:
            stored_dim = self._get_vector_store().stored_vector_dim()
            embed_dim = self._embedder.dimensions
            if stored_dim is not None and stored_dim != embed_dim:
                logger.warning(
                    "Embedder dimension (%d) differs from stored index (%d) — "
                    "rebuilding index requires --force to avoid mixed dimensions. "
                    "Skipping vector update.",
                    embed_dim, stored_dim,
                )
                embedding_failed = True

        batch_size = self._settings.embedding_batch_size
        all_embeddings: list[list[float]] = []
        texts = [text for _, text in rows_to_embed]
        if not embedding_failed:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                try:
                    embeddings = self._embedder.embed_batch(batch)
                    # Validate dimension consistency within batch
                    if all_embeddings and len(embeddings[0]) != len(all_embeddings[0]):
                        logger.warning(
                            "Embedding dimension changed mid-batch (%d -> %d) — "
                            "aborting vector index to prevent corruption",
                            len(all_embeddings[0]), len(embeddings[0]),
                        )
                        embedding_failed = True
                        all_embeddings.clear()
                        break
                    all_embeddings.extend(embeddings)
                    logger.info("Embedded batch %d-%d", i, i + len(batch))
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                        logger.warning(
                            "Gemini embedding quota exhausted at batch %d — "
                            "skipping vector index update, BM25 will still be rebuilt",
                            i,
                        )
                        embedding_failed = True
                        all_embeddings.clear()
                        break
                    raise

        # Step 4: Build rows and upsert (skip if embedding failed or dim mismatch)
        if not embedding_failed and all_embeddings:
            actual_dim = len(all_embeddings[0])
            vector_rows = []
            for idx, (chunk, combined) in enumerate(rows_to_embed):
                row = {
                    "vector": all_embeddings[idx],
                    "video_id": chunk.video_id,
                    "chunk_index": chunk.chunk_index,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "duration": chunk.duration,
                    "combined_text": combined,
                    "volume_level": compute_volume_level(
                        chunk, all_chunks, self._settings.raised_voice_stddev_threshold
                    ),
                    "has_speech": bool(chunk.transcript),  # D-06
                    "has_ocr": bool(chunk.ocr_results),     # D-06
                    "has_raised_voice": (
                        chunk.audio_features.has_raised_voice
                        if chunk.audio_features else False
                    ),  # D-06
                    "scene_type": chunk.scene_type,
                }
                vector_rows.append(row)

            if vector_rows:
                self._get_vector_store(vector_dim=actual_dim).upsert(vector_rows)
                logger.info("Upserted %d rows to vector store", len(vector_rows))
        else:
            logger.info("Vector store unchanged (embedding quota exhausted)")

        # Step 5: Build BM25 index — always over the FULL corpus.
        # BM25 is rebuilt from scratch each time, so it must include every
        # ingested video, not just the ones requested in this call. Otherwise
        # `index video_b` would silently drop video_a from keyword search while
        # its vectors remain in LanceDB, breaking hybrid search consistency.
        all_metadata_ids = [
            p.stem for p in self._settings.metadata_dir.glob("*.json")
        ]
        bm25_chunks: list[ChunkMetadata] = []
        for vid in all_metadata_ids:
            bm25_chunks.extend(self._metadata_writer.load(vid))
        self._bm25_store.build(bm25_chunks)
        bm25_path = self._settings.index_dir / "bm25.pkl"
        self._bm25_store.save(str(bm25_path))
        logger.info(
            "Built and saved BM25 index over %d videos at %s",
            len(all_metadata_ids), bm25_path,
        )

        return {
            "total_chunks": len(all_chunks),
            "embedded_chunks": len(rows_to_embed),
            "skipped_chunks": len(all_chunks) - len(rows_to_embed),
            "skipped_videos": len(skipped_videos),
            "bm25_indexed": self._bm25_store._corpus_size,
        }
