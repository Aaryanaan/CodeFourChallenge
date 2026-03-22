"""Data models for video search pipeline."""

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata for a single video chunk.

    Represents a segment of a video identified by scene detection
    or sliding window fallback. Used as the core data contract
    throughout the pipeline.
    """

    video_id: str
    chunk_index: int
    start_time: float
    end_time: float
    duration: float
    scene_type: str  # "detected" or "sliding_window"
