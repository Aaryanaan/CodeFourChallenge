"""GeminiCaptioner: visual captioning via Gemini Flash with disk cache.

Implements the Captioner protocol. Uses a cache-first flow: checks disk
before making any API call. On cache miss, extracts a clip via ffmpeg,
uploads to Google Files API, calls generate_content, then cleans up.
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from google import genai

from videosearch.config import Settings


CAPTION_SYSTEM_PROMPT = """You are analyzing body-worn camera footage for a law enforcement video search system.

For the video clip provided, describe EXACTLY these 5 fields with specific, concrete details.
Be precise -- say "dark navy uniform with gold badge on left chest" not "police uniform".
Say "white sedan, possibly Honda Civic, license plate partially visible" not "car".

Clothing: Describe all visible clothing items with colors, patterns, and distinguishing features.
Actions: Describe specific physical actions occurring -- who is doing what, movements, gestures.
Objects: List visible objects with descriptions -- vehicles (make/color/type), weapons, tools, signs, furniture.
Text: Transcribe any visible text -- signs, license plates, badges, documents. Write 'none' if no text visible.
Lighting: Describe lighting conditions -- time of day, light sources, shadows, visibility quality."""

CAPTION_USER_PROMPT = """Analyze this body camera footage clip. Respond with ONLY the 5 labeled fields, no preamble:

Clothing:
Actions:
Objects:
Text:
Lighting:"""


class GeminiCaptioner:
    """Visual captioner using Gemini Flash via Google GenAI SDK.

    Satisfies the Captioner protocol. Uses disk cache to avoid redundant
    API calls. Handles ffmpeg clip extraction and Files API lifecycle
    (upload + delete) with cleanup even on error.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(api_key=settings.google_api_key)
        # Strip OpenRouter prefix if present (google/ prefix is not valid for native SDK)
        self._model = settings.captioner_model.replace("google/", "")
        self._cache_dir = settings.caption_cache_dir
        self._ffmpeg_path = settings.ffmpeg_path

    def caption(self, video_path: str, start: float, end: float) -> dict:
        """Extract a visual caption for a video chunk.

        Args:
            video_path: Path to the source video file.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            Dict with keys: "caption" (str), "cached" (bool).
        """
        video_id = Path(video_path).stem.replace("_720p", "")
        chunk_index = int(start)

        # Cache-first: return cached value without any API call
        cached_text = self._load_cache(video_id, chunk_index)
        if cached_text is not None:
            return {"caption": cached_text, "cached": True}

        # Extract clip to temp file
        clip_path = self._extract_clip(video_path, start, end)
        try:
            # Upload and caption
            uploaded = self._client.files.upload(file=clip_path)
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[CAPTION_SYSTEM_PROMPT, uploaded, CAPTION_USER_PROMPT],
                )
                caption_text = response.text
            finally:
                # Always delete uploaded file
                self._client.files.delete(name=uploaded.name)
        finally:
            # Always delete temp clip
            os.unlink(clip_path)

        self._save_cache(video_id, chunk_index, caption_text)
        return {"caption": caption_text, "cached": False}

    def _extract_clip(self, video_path: str, start: float, end: float) -> str:
        """Extract a video clip using ffmpeg.

        Args:
            video_path: Path to source video.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            Path to the extracted temp clip (caller is responsible for deletion).
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        subprocess.run(
            [
                self._ffmpeg_path,
                "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(end - start),
                "-c", "copy",
                "-an",
                tmp_path,
            ],
            check=True,
            capture_output=True,
        )
        return tmp_path

    def _cache_path(self, video_id: str, chunk_index: int) -> Path:
        """Return the cache file path for a given video/chunk."""
        return Path(self._cache_dir) / video_id / f"{chunk_index}.json"

    def _load_cache(self, video_id: str, chunk_index: int) -> Optional[str]:
        """Load a cached caption if it exists.

        Returns:
            Cached caption string, or None if no cache entry.
        """
        path = self._cache_path(video_id, chunk_index)
        if path.exists():
            data = json.loads(path.read_text())
            return data["caption"]
        return None

    def _save_cache(self, video_id: str, chunk_index: int, caption: str) -> None:
        """Write a caption to the disk cache.

        Creates parent directories as needed. Writes JSON with keys:
        "caption", "model", "cached_at".
        """
        path = self._cache_path(video_id, chunk_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "caption": caption,
            "model": self._model,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(cache_data))
