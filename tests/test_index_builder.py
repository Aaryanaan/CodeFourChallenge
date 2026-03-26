"""Tests for IndexBuilder orchestrator and helper functions."""

import pytest

from videosearch.models import (
    AudioFeatures,
    ChunkMetadata,
    OCRResult,
    TranscriptSegment,
)


def _make_audio(rms_mean: float = 0.1) -> AudioFeatures:
    return AudioFeatures(
        rms_mean=rms_mean,
        rms_max=rms_mean * 2,
        rms_stddev=0.01,
        zcr_mean=0.05,
        zcr_max=0.1,
        zcr_stddev=0.01,
        has_raised_voice=False,
    )


def _make_chunk(
    video_id: str = "v1",
    chunk_index: int = 0,
    transcript_texts: list[str] | None = None,
    ocr_texts: list[str] | None = None,
    rms_mean: float = 0.1,
    has_audio: bool = True,
) -> ChunkMetadata:
    transcript = None
    if transcript_texts is not None:
        transcript = [
            TranscriptSegment(
                text=t, start=0.0, end=1.0, avg_logprob=-0.3, words=[]
            )
            for t in transcript_texts
        ]
    ocr_results = None
    if ocr_texts is not None:
        ocr_results = [
            OCRResult(
                text=t,
                confidence=0.95,
                first_seen=0.0,
                last_seen=1.0,
                bbox=[[0, 0], [1, 0], [1, 1], [0, 1]],
            )
            for t in ocr_texts
        ]
    audio = _make_audio(rms_mean) if has_audio else None
    return ChunkMetadata(
        video_id=video_id,
        chunk_index=chunk_index,
        start_time=0.0,
        end_time=30.0,
        duration=30.0,
        scene_type="detected",
        transcript=transcript,
        audio_features=audio,
        ocr_results=ocr_results,
    )


# --- build_combined_text tests ---

class TestBuildCombinedText:
    def test_combined_text_format(self):
        """Chunk with both transcript and OCR returns labeled format."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk(transcript_texts=["hello", "world"], ocr_texts=["STOP SIGN"])
        result = build_combined_text(chunk)
        assert result == "Transcript: hello world\nOCR: STOP SIGN"

    def test_combined_text_transcript_only(self):
        """Chunk with only transcript returns transcript section."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk(transcript_texts=["hello", "world"])
        result = build_combined_text(chunk)
        assert result == "Transcript: hello world"

    def test_combined_text_ocr_only(self):
        """Chunk with only OCR returns OCR section."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk(ocr_texts=["STOP"])
        result = build_combined_text(chunk)
        assert result == "OCR: STOP"

    def test_combined_text_empty(self):
        """Chunk with neither transcript nor OCR returns empty string."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk()
        result = build_combined_text(chunk)
        assert result == ""


# --- skip empty chunks test ---

class TestSkipEmptyChunks:
    def test_skip_empty_chunks(self):
        """Chunks with no transcript AND no OCR are excluded from embeddable set."""
        from videosearch.index_builder import build_combined_text

        chunks = [
            _make_chunk(chunk_index=0, transcript_texts=["hello"]),
            _make_chunk(chunk_index=1),  # empty — no transcript, no OCR
            _make_chunk(chunk_index=2, ocr_texts=["EXIT"]),
        ]
        embeddable = [c for c in chunks if build_combined_text(c)]
        assert len(embeddable) == 2
        assert embeddable[0].chunk_index == 0
        assert embeddable[1].chunk_index == 2


# --- compute_volume_level tests ---

class TestComputeVolumeLevel:
    def test_volume_level_bins(self):
        """Volume level is quiet/normal/loud based on RMS distribution."""
        from videosearch.index_builder import compute_volume_level

        # Create a distribution with tight clustering around 0.1 so outliers
        # are clearly beyond 2 stddevs. 100 chunks at 0.1 gives mean~0.1,
        # stddev~0.0 so 0.01 is clearly quiet and 0.5 is clearly loud.
        chunks = [
            _make_chunk(chunk_index=i, rms_mean=0.1) for i in range(100)
        ]
        quiet_chunk = _make_chunk(chunk_index=100, rms_mean=0.01)
        loud_chunk = _make_chunk(chunk_index=101, rms_mean=0.5)
        all_chunks = chunks + [quiet_chunk, loud_chunk]

        assert compute_volume_level(quiet_chunk, all_chunks) == "quiet"
        assert compute_volume_level(loud_chunk, all_chunks) == "loud"
        assert compute_volume_level(chunks[0], all_chunks) == "normal"


# --- boolean flag tests ---

class TestBooleanFlags:
    def test_has_speech_flag(self):
        """Chunk with transcript has truthy transcript field."""
        chunk_with = _make_chunk(transcript_texts=["hello"])
        chunk_without = _make_chunk()
        assert bool(chunk_with.transcript)
        assert not chunk_without.transcript

    def test_has_ocr_flag(self):
        """Chunk with OCR results has truthy ocr_results field."""
        chunk_with = _make_chunk(ocr_texts=["STOP"])
        chunk_without = _make_chunk()
        assert bool(chunk_with.ocr_results)
        assert not chunk_without.ocr_results


# --- build_combined_text caption tests (D-23) ---

class TestBuildCombinedTextWithCaption:
    def test_combined_text_with_caption(self):
        """Chunk with transcript + OCR + caption includes all three sections."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk(transcript_texts=["hello"], ocr_texts=["STOP"])
        chunk.visual_caption = "Clothing: navy uniform\nActions: walking"
        result = build_combined_text(chunk)
        assert result == "Transcript: hello\nOCR: STOP\nCaption: Clothing: navy uniform\nActions: walking"

    def test_combined_text_caption_only(self):
        """Chunk with only visual_caption returns Caption section."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk()
        chunk.visual_caption = "Clothing: navy uniform"
        result = build_combined_text(chunk)
        assert result == "Caption: Clothing: navy uniform"

    def test_combined_text_no_caption(self):
        """Chunk without visual_caption produces no Caption section (backward compat)."""
        from videosearch.index_builder import build_combined_text

        chunk = _make_chunk(transcript_texts=["hello"])
        result = build_combined_text(chunk)
        assert "Caption:" not in result


# --- Incremental indexing tests (IDX-05, D-05, D-07) ---

class TestIncrementalIndexing:
    def test_build_index_skips_indexed_video(self):
        """When count_by_video matches chunk count, embedder.embed_batch is NOT called."""
        from unittest.mock import MagicMock, patch

        chunks_v1 = [_make_chunk(video_id="v1", chunk_index=i, transcript_texts=["hello"]) for i in range(3)]

        mock_settings = MagicMock()
        mock_settings.index_dir = "/tmp/test_idx"
        mock_settings.metadata_dir = "/tmp/test_meta"
        mock_settings.embedding_batch_size = 50
        mock_settings.raised_voice_stddev_threshold = 2.0

        with patch("videosearch.index_builder.MetadataWriter") as MockWriter, \
             patch("videosearch.index_builder.GeminiEmbedder") as MockEmbedder, \
             patch("videosearch.index_builder.LanceVectorStore") as MockVStore, \
             patch("videosearch.index_builder.BM25Store") as MockBM25:

            mock_writer = MockWriter.return_value
            mock_writer.load.return_value = chunks_v1

            mock_vstore = MockVStore.return_value
            # count_by_video returns 3 = matches len(chunks_v1) => should skip
            mock_vstore.count_by_video.return_value = 3

            mock_embedder = MockEmbedder.return_value
            mock_bm25 = MockBM25.return_value
            mock_bm25._corpus_size = 0

            # Mock metadata_dir.glob for BM25 rebuild
            mock_settings.metadata_dir.glob.return_value = []

            from videosearch.index_builder import IndexBuilder
            builder = IndexBuilder.__new__(IndexBuilder)
            builder._settings = mock_settings
            builder._embedder = mock_embedder
            builder._vector_store = mock_vstore
            builder._bm25_store = mock_bm25
            builder._metadata_writer = mock_writer

            stats = builder.build_index(["v1"])

            # Embedder should NOT have been called since video was skipped
            mock_embedder.embed_batch.assert_not_called()
            assert stats["skipped_videos"] == 1

    def test_build_index_force_reembeds(self):
        """When force=True, embedder.embed_batch IS called even for already-indexed video."""
        from unittest.mock import MagicMock, patch

        chunks_v1 = [_make_chunk(video_id="v1", chunk_index=i, transcript_texts=["hello"]) for i in range(3)]

        mock_settings = MagicMock()
        mock_settings.index_dir = "/tmp/test_idx"
        mock_settings.metadata_dir = "/tmp/test_meta"
        mock_settings.embedding_batch_size = 50
        mock_settings.raised_voice_stddev_threshold = 2.0

        with patch("videosearch.index_builder.MetadataWriter") as MockWriter, \
             patch("videosearch.index_builder.GeminiEmbedder") as MockEmbedder, \
             patch("videosearch.index_builder.LanceVectorStore") as MockVStore, \
             patch("videosearch.index_builder.BM25Store") as MockBM25:

            mock_writer = MockWriter.return_value
            mock_writer.load.return_value = chunks_v1

            mock_vstore = MockVStore.return_value
            # count_by_video returns 3 = matches, but force=True should override
            mock_vstore.count_by_video.return_value = 3

            mock_embedder = MockEmbedder.return_value
            mock_embedder.embed_batch.return_value = [[0.1] * 768 for _ in range(3)]
            mock_bm25 = MockBM25.return_value
            mock_bm25._corpus_size = 3

            mock_settings.metadata_dir.glob.return_value = []

            from videosearch.index_builder import IndexBuilder
            builder = IndexBuilder.__new__(IndexBuilder)
            builder._settings = mock_settings
            builder._embedder = mock_embedder
            builder._vector_store = mock_vstore
            builder._bm25_store = mock_bm25
            builder._metadata_writer = mock_writer

            stats = builder.build_index(["v1"], force=True)

            # Embedder SHOULD have been called despite matching count
            mock_embedder.embed_batch.assert_called()
            assert stats.get("skipped_videos", 0) == 0
