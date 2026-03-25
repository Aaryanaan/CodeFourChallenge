"""Data models for video search pipeline."""

from typing import Optional

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech from a video chunk.

    Produced by faster-whisper with word-level timestamps.
    """

    text: str
    start: float
    end: float
    avg_logprob: float
    words: list[dict]  # Each dict: {"word": str, "start": float, "end": float, "probability": float}


class AudioFeatures(BaseModel):
    """Acoustic features extracted from a video chunk via librosa.

    Pitch fields are Optional because they are None when all frames
    are unvoiced (no fundamental frequency detected).
    """

    rms_mean: float
    rms_max: float
    rms_stddev: float
    pitch_mean: Optional[float] = None
    pitch_max: Optional[float] = None
    pitch_stddev: Optional[float] = None
    zcr_mean: float
    zcr_max: float
    zcr_stddev: float
    has_raised_voice: bool


class OCRResult(BaseModel):
    """Text detected in video frames via PaddleOCR.

    Tracks the text, confidence, temporal span, and bounding box
    across sampled frames.
    """

    text: str
    confidence: float
    first_seen: float
    last_seen: float
    bbox: list[list[float]]


class ChunkMetadata(BaseModel):
    """Metadata for a single video chunk.

    Represents a segment of a video identified by scene detection
    or sliding window fallback. Used as the core data contract
    throughout the pipeline. Optional extraction fields are populated
    by downstream extractors (Transcriber, AudioAnalyzer, OCRExtractor).
    """

    video_id: str
    chunk_index: int
    start_time: float
    end_time: float
    duration: float
    scene_type: str  # "detected" or "sliding_window"

    # Extraction results (populated by downstream extractors)
    transcript: Optional[list[TranscriptSegment]] = None
    audio_features: Optional[AudioFeatures] = None
    ocr_results: Optional[list[OCRResult]] = None
    visual_caption: Optional[str] = None


class SearchResult(BaseModel):
    """A single search result with provenance."""

    video_id: str
    chunk_index: int
    start_time: float
    end_time: float
    score: float
    transcript_snippet: str = ""
    reasoning: str = ""  # Phase 6 fills via LLM reranker
