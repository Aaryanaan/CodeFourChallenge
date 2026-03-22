"""Scene-aware video chunking with sliding window fallback.

Uses PySceneDetect AdaptiveDetector for scene boundary detection,
then enforces min/max duration bounds. Falls back to sliding window
if no scenes are detected. Does NOT save individual clip files (per D-07).
"""

from pathlib import Path

from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector

from videosearch.models import ChunkMetadata


class SceneAwareChunker:
    """Scene-aware video chunker with sliding window fallback.

    Uses PySceneDetect AdaptiveDetector for scene boundary detection,
    then enforces min/max duration bounds. Falls back to sliding window
    if no scenes are detected.
    """

    def __init__(
        self,
        threshold: float = 3.0,
        min_duration: float = 10.0,
        max_duration: float = 60.0,
        window_size: float = 30.0,
        window_overlap: float = 10.0,
    ):
        self.threshold = threshold
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.window_size = window_size
        self.window_overlap = window_overlap

    def chunk(self, video_path: str) -> list[ChunkMetadata]:
        """Split video into scene-aware chunks.

        Returns list of ChunkMetadata with start/end timestamps.
        Does NOT extract individual clip files (per D-07).
        """
        video = open_video(video_path)
        video_duration = video.duration.get_seconds()
        video_id = Path(video_path).stem

        # Detect scenes
        scene_manager = SceneManager()
        scene_manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=self.threshold,
                min_scene_len=15,  # minimum frames per scene
            )
        )
        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()

        if not scenes:
            # Fallback: pure sliding window
            raw_chunks = self._sliding_window(0.0, video_duration)
            scene_type = "sliding_window"
        else:
            raw_chunks = [
                (s[0].get_seconds(), s[1].get_seconds())
                for s in scenes
            ]
            raw_chunks = self._enforce_bounds(raw_chunks, video_duration)
            scene_type = "detected"

        # Convert to ChunkMetadata
        return [
            ChunkMetadata(
                video_id=video_id,
                chunk_index=i,
                start_time=round(start, 3),
                end_time=round(end, 3),
                duration=round(end - start, 3),
                scene_type=scene_type,
            )
            for i, (start, end) in enumerate(raw_chunks)
        ]

    def _sliding_window(
        self, start: float, end: float
    ) -> list[tuple[float, float]]:
        """Generate sliding window chunks over a time range."""
        chunks: list[tuple[float, float]] = []
        pos = start
        stride = self.window_size - self.window_overlap

        while pos < end:
            chunk_end = min(pos + self.window_size, end)
            # Only include if chunk meets min duration or it is the last chunk
            if chunk_end - pos >= self.min_duration or chunk_end == end:
                chunks.append((pos, chunk_end))
            if chunk_end == end:
                break
            pos += stride

        # Handle edge case: if last chunk is too short, merge with previous
        if (
            len(chunks) > 1
            and (chunks[-1][1] - chunks[-1][0]) < self.min_duration
        ):
            prev = chunks.pop(-2)
            last = chunks.pop(-1)
            merged = (prev[0], last[1])
            # If merged chunk exceeds max, keep them separate
            if merged[1] - merged[0] <= self.max_duration:
                chunks.append(merged)
            else:
                chunks.append(prev)
                chunks.append(last)

        return chunks

    def _enforce_bounds(
        self,
        chunks: list[tuple[float, float]],
        video_duration: float,
    ) -> list[tuple[float, float]]:
        """Merge short chunks and split long chunks to enforce duration bounds."""
        # Phase 1: Merge chunks shorter than min_duration with neighbors
        merged: list[tuple[float, float]] = []
        for start, end in chunks:
            duration = end - start
            if merged and (duration < self.min_duration):
                # Merge with previous chunk
                prev_start, _ = merged[-1]
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        # Check if last chunk is too short and merge backwards
        if len(merged) > 1:
            last_start, last_end = merged[-1]
            if last_end - last_start < self.min_duration:
                prev_start, _ = merged[-2]
                merged[-2] = (prev_start, last_end)
                merged.pop(-1)

        # Phase 2: Split chunks longer than max_duration using sliding window
        result: list[tuple[float, float]] = []
        for start, end in merged:
            duration = end - start
            if duration > self.max_duration:
                result.extend(self._sliding_window(start, end))
            else:
                result.append((start, end))

        return result
