"""Tests for SceneAwareChunker -- scene-aware chunking with sliding window fallback."""

import pytest

from videosearch.chunker import SceneAwareChunker
from videosearch.models import ChunkMetadata
from videosearch.protocols import Chunker


class TestChunkerProtocol:
    """Verify SceneAwareChunker satisfies the Chunker protocol."""

    def test_chunker_satisfies_protocol(self):
        chunker = SceneAwareChunker()
        assert isinstance(chunker, Chunker)

    def test_chunker_has_chunk_method(self):
        chunker = SceneAwareChunker()
        assert hasattr(chunker, "chunk")
        assert callable(chunker.chunk)


class TestChunkResults:
    """Integration tests for scene-aware chunking on a long video."""

    def test_chunk_returns_chunk_metadata(self, test_video_long):
        """Verify returned objects are ChunkMetadata instances."""
        chunker = SceneAwareChunker()
        chunks = chunker.chunk(test_video_long)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, ChunkMetadata)

    def test_chunk_duration_bounds(self, test_video_long):
        """All chunks have 10.0 <= duration <= 60.0."""
        chunker = SceneAwareChunker()
        chunks = chunker.chunk(test_video_long)
        assert len(chunks) > 0
        for chunk in chunks:
            assert 10.0 <= chunk.duration <= 60.0, (
                f"Chunk {chunk.chunk_index} duration {chunk.duration}s "
                f"outside bounds [10, 60]"
            )

    def test_chunk_sequential_indices(self, test_video_long):
        """chunk_index values are 0, 1, 2, ... sequential."""
        chunker = SceneAwareChunker()
        chunks = chunker.chunk(test_video_long)
        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i, (
                f"Expected chunk_index={i}, got {chunk.chunk_index}"
            )

    def test_chunk_coverage(self, test_video_long):
        """First chunk starts at 0.0, last chunk ends near video duration."""
        chunker = SceneAwareChunker()
        chunks = chunker.chunk(test_video_long)
        assert len(chunks) > 0
        assert chunks[0].start_time == 0.0, (
            f"First chunk starts at {chunks[0].start_time}, expected 0.0"
        )
        # Last chunk should end within 1.0s of video duration (120s)
        assert chunks[-1].end_time >= 119.0, (
            f"Last chunk ends at {chunks[-1].end_time}, expected near 120.0"
        )

    def test_chunk_no_gaps(self, test_video_long):
        """Consecutive chunks have no gaps (within 0.5s tolerance)."""
        chunker = SceneAwareChunker()
        chunks = chunker.chunk(test_video_long)
        assert len(chunks) > 1
        for i in range(len(chunks) - 1):
            gap = abs(chunks[i].end_time - chunks[i + 1].start_time)
            assert gap < 0.5, (
                f"Gap of {gap}s between chunk {i} (end={chunks[i].end_time}) "
                f"and chunk {i+1} (start={chunks[i+1].start_time})"
            )

    def test_chunk_scene_type_valid(self, test_video_long):
        """All chunks have scene_type in ('detected', 'sliding_window')."""
        chunker = SceneAwareChunker()
        chunks = chunker.chunk(test_video_long)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.scene_type in ("detected", "sliding_window"), (
                f"Invalid scene_type: {chunk.scene_type}"
            )

    def test_chunk_video_id(self, test_video_long):
        """video_id matches the stem of the input filename."""
        chunker = SceneAwareChunker()
        chunks = chunker.chunk(test_video_long)
        assert len(chunks) > 0
        from pathlib import Path
        expected_id = Path(test_video_long).stem
        for chunk in chunks:
            assert chunk.video_id == expected_id


class TestSlidingWindowFallback:
    """Test sliding window fallback when no scenes detected."""

    def test_sliding_window_fallback(self, test_video):
        """With very high threshold, no scenes detected -- falls back to sliding window."""
        chunker = SceneAwareChunker(threshold=100.0)
        chunks = chunker.chunk(test_video)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.scene_type == "sliding_window"
            assert chunk.duration >= 10.0 or chunk == chunks[-1]
