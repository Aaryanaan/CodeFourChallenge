"""Librosa-based audio feature extraction with raised-voice detection.

Extracts RMS energy, pitch (pYIN), and zero-crossing rate from video
audio segments. Satisfies the AudioAnalyzer protocol structurally.

Raised-voice detection uses per-video relative thresholds: a chunk is
flagged when its RMS exceeds mean + N*stddev across all video chunks.
This is a two-pass design -- analyze() sets has_raised_voice=False,
and the caller invokes detect_raised_voice() after all chunks are processed.
"""

from pathlib import Path

import librosa
import numpy as np

from videosearch.audio_utils import extract_audio_segment


class LibrosaAudioAnalyzer:
    """Extract audio features using librosa. Satisfies AudioAnalyzer protocol.

    Features extracted:
    - RMS energy (mean, max, stddev)
    - Pitch via pYIN (mean, max, stddev) -- None when no voiced frames
    - Zero-crossing rate (mean, max, stddev)
    - has_raised_voice placeholder (False; set by caller via detect_raised_voice)
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def analyze(self, video_path: str, start: float, end: float) -> dict:
        """Extract audio features from a video segment.

        Args:
            video_path: Path to the source video file.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            Dict matching AudioFeatures fields: rms_mean, rms_max, rms_stddev,
            pitch_mean, pitch_max, pitch_stddev, zcr_mean, zcr_max, zcr_stddev,
            has_raised_voice.
        """
        audio_path = extract_audio_segment(
            video_path, start, end, ffmpeg_path=self.ffmpeg_path
        )
        try:
            # Load audio preserving original sample rate (16kHz from extraction)
            y, sr = librosa.load(audio_path, sr=None)

            # RMS energy (D-07)
            rms = librosa.feature.rms(y=y)[0]

            # Pitch via pYIN (D-07) -- returns NaN for unvoiced frames
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )

            # Zero-crossing rate (D-07)
            zcr = librosa.feature.zero_crossing_rate(y=y)[0]

            # Build features dict with NaN-safe pitch handling (Pitfall 3)
            has_voiced = np.any(voiced_flag) if voiced_flag is not None else False

            return {
                "rms_mean": float(np.mean(rms)),
                "rms_max": float(np.max(rms)),
                "rms_stddev": float(np.std(rms)),
                "pitch_mean": float(np.nanmean(f0)) if has_voiced else None,
                "pitch_max": float(np.nanmax(f0)) if has_voiced else None,
                "pitch_stddev": float(np.nanstd(f0)) if has_voiced else None,
                "zcr_mean": float(np.mean(zcr)),
                "zcr_max": float(np.max(zcr)),
                "zcr_stddev": float(np.std(zcr)),
                "has_raised_voice": False,  # Caller sets via detect_raised_voice
            }
        finally:
            Path(audio_path).unlink(missing_ok=True)

    @staticmethod
    def detect_raised_voice(
        chunk_rms_max: float,
        video_rms_values: list[float],
        stddev_threshold: float = 2.0,
    ) -> bool:
        """Determine if a chunk has raised voice using per-video relative thresholds.

        A chunk is flagged when its RMS max exceeds the video-level mean + N*stddev
        (D-05). Must be called after all chunks of a video are analyzed.

        Args:
            chunk_rms_max: RMS max of the chunk being evaluated.
            video_rms_values: List of rms_max values from all chunks of the video.
            stddev_threshold: Number of standard deviations above mean (default 2.0).

        Returns:
            True if chunk_rms_max exceeds the threshold.
        """
        video_mean = float(np.mean(video_rms_values))
        video_std = float(np.std(video_rms_values))
        return bool(chunk_rms_max > video_mean + stddev_threshold * video_std)
