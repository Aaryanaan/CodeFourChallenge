"""Whisper-based audio transcription with VAD and hallucination filtering.

Uses faster-whisper to transcribe audio segments extracted from video files.
Satisfies the Transcriber protocol structurally (no inheritance required).
"""

from pathlib import Path

from faster_whisper import WhisperModel

from videosearch.audio_utils import extract_audio_segment


class WhisperTranscriber:
    """Transcribe audio using faster-whisper. Satisfies Transcriber protocol.

    Applies two quality filters:
    - VAD filter (built into faster-whisper) to skip non-speech regions
    - Hallucination filter: drops segments with avg_logprob < -1.0
    - No-speech filter: drops segments with no_speech_prob > 0.6

    Temp audio files are cleaned up in a finally block to prevent leaks.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        compute_type: str = "auto",
        ffmpeg_path: str = "ffmpeg",
    ):
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        self.ffmpeg_path = ffmpeg_path

    def transcribe(self, video_path: str, start: float, end: float) -> dict:
        """Transcribe audio segment from video with word-level timestamps.

        Args:
            video_path: Path to the source video file.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            Dict with keys:
                - segments: list of segment dicts (text, start, end, avg_logprob, words)
                - language: detected language code
                - language_probability: confidence in language detection
        """
        audio_path = extract_audio_segment(
            video_path, start, end, ffmpeg_path=self.ffmpeg_path
        )
        try:
            segments_iter, info = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                language="en",
            )

            filtered_segments = []
            for segment in segments_iter:
                # Hallucination filter (D-03): skip low-confidence segments
                if segment.avg_logprob < -1.0:
                    continue
                # No-speech guard: skip segments likely containing no speech
                if segment.no_speech_prob > 0.6:
                    continue

                seg_dict = {
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end,
                    "avg_logprob": segment.avg_logprob,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in (segment.words or [])
                    ],
                }
                filtered_segments.append(seg_dict)

            return {
                "segments": filtered_segments,
                "language": info.language,
                "language_probability": info.language_probability,
            }
        finally:
            Path(audio_path).unlink(missing_ok=True)
