"""Ingestion pipeline orchestrator for video processing.

Wires together all Phase 1-2 components into a single ingest() call:
  1. Compress video to 720p (FFmpegCompressor)
  2. Chunk into scene-aware segments (SceneAwareChunker)
  3. Extract per chunk: transcribe, audio features, OCR (three extractors)
  4. Two-pass raised voice detection (LibrosaAudioAnalyzer.detect_raised_voice)
  5. Write metadata to JSON (MetadataWriter)

Extractor failures are logged as warnings and do not abort the pipeline.
"""

import logging
from pathlib import Path

from videosearch.audio_analyzer import LibrosaAudioAnalyzer
from videosearch.chunker import SceneAwareChunker
from videosearch.compressor import FFmpegCompressor
from videosearch.config import Settings
from videosearch.metadata_writer import MetadataWriter
from videosearch.models import AudioFeatures, ChunkMetadata, OCRResult, TranscriptSegment
from videosearch.ocr_extractor import PaddleOCRExtractor
from videosearch.transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrate full video ingestion: compress, chunk, extract, index metadata.

    Each component is hotswappable via constructor injection. The pipeline
    reads settings once at init and passes relevant parameters to each component.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._compressor = FFmpegCompressor(settings.ffmpeg_path)
        self._chunker = SceneAwareChunker(
            threshold=settings.pyscenedetect_threshold,
            min_duration=settings.chunk_min_duration,
            max_duration=settings.chunk_max_duration,
            window_size=settings.sliding_window_size,
            window_overlap=settings.sliding_window_overlap,
        )
        self._transcriber = WhisperTranscriber(
            model_size=settings.whisper_model,
            compute_type=settings.whisper_compute_type,
            ffmpeg_path=settings.ffmpeg_path,
        )
        self._audio_analyzer = LibrosaAudioAnalyzer(ffmpeg_path=settings.ffmpeg_path)
        self._ocr_extractor = PaddleOCRExtractor(
            confidence_threshold=settings.ocr_confidence_threshold,
            frame_interval=settings.ocr_frame_interval,
        )
        self._metadata_writer = MetadataWriter(metadata_dir=settings.metadata_dir)

    def ingest(self, video_path: str, include_caption: bool = False) -> str:
        """Ingest a video through the full processing pipeline.

        Steps:
          1. Compress to 720p h264
          2. Detect scenes and chunk
          3. Run transcription, audio analysis, and OCR per chunk
          4. Two-pass raised voice detection
          5. Write metadata JSON
          6. (Optional) Generate visual captions if include_caption=True

        Args:
            video_path: Absolute or relative path to the source video.
            include_caption: If True, run visual captioning after extraction.

        Returns:
            video_id derived from the video filename stem (e.g. "bodycam_001").
        """
        video_id = Path(video_path).stem

        # Step 1: Compress
        compressed_dir = self._settings.video_dir / "compressed"
        compressed_dir.mkdir(parents=True, exist_ok=True)
        compressed_path = str(compressed_dir / f"{video_id}_720p.mp4")
        self._compressor.compress(video_path, compressed_path)

        # Step 2: Chunk
        chunks = self._chunker.chunk(compressed_path)

        # Step 3: Extract per chunk
        for chunk in chunks:
            self._extract_chunk(chunk, compressed_path)

        # Step 4: Two-pass raised voice detection
        video_rms_values = [
            c.audio_features.rms_max
            for c in chunks
            if c.audio_features is not None
        ]
        for chunk in chunks:
            if chunk.audio_features is not None:
                chunk.audio_features.has_raised_voice = (
                    LibrosaAudioAnalyzer.detect_raised_voice(
                        chunk.audio_features.rms_max,
                        video_rms_values,
                        self._settings.raised_voice_stddev_threshold,
                    )
                )

        # Step 5: Write metadata
        self._metadata_writer.write(video_id, chunks)

        # Step 6: Optional visual captioning (D-02)
        if include_caption:
            from videosearch.captioner import GeminiCaptioner

            captioner = GeminiCaptioner(self._settings)
            for chunk in chunks:
                try:
                    result = captioner.caption(
                        compressed_path, chunk.start_time, chunk.end_time, chunk.chunk_index
                    )
                    chunk.visual_caption = result["caption"]
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Caption failed for chunk %d: %s", chunk.chunk_index, e
                    )
            # Re-write metadata with captions
            self._metadata_writer.write(video_id, chunks)

        return video_id

    def _extract_chunk(self, chunk: ChunkMetadata, video_path: str) -> None:
        """Run all three extractors on a single chunk.

        Each extractor is tried independently. Failures are logged as warnings
        and do not abort extraction of other modalities or subsequent chunks.

        Args:
            chunk: ChunkMetadata to populate (mutated in place).
            video_path: Path to the compressed video file.
        """
        # Transcription
        try:
            result = self._transcriber.transcribe(
                video_path, chunk.start_time, chunk.end_time
            )
            chunk.transcript = [
                TranscriptSegment(**seg) for seg in result["segments"]
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Transcription failed for chunk %d: %s", chunk.chunk_index, e
            )

        # Audio features
        try:
            result = self._audio_analyzer.analyze(
                video_path, chunk.start_time, chunk.end_time
            )
            chunk.audio_features = AudioFeatures(**result)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Audio analysis failed for chunk %d: %s", chunk.chunk_index, e
            )

        # OCR
        try:
            result = self._ocr_extractor.extract(
                video_path, chunk.start_time, chunk.end_time
            )
            chunk.ocr_results = [OCRResult(**r) for r in result["results"]]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "OCR extraction failed for chunk %d: %s", chunk.chunk_index, e
            )
