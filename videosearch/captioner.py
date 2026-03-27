"""GeminiCaptioner: visual captioning via OpenRouter with disk cache.

Uses OpenRouter chat completions API with base64-encoded video clips.
Cache-first flow: checks disk before making any API call. On cache miss,
extracts a clip via ffmpeg, sends to OpenRouter, then cleans up.
"""

import base64
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from videosearch.config import Settings

logger = logging.getLogger(__name__)


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


class QuotaExhaustedError(Exception):
    """Raised when the API quota is exhausted -- caller should stop sending requests."""


class GeminiCaptioner:
    """Visual captioner using OpenRouter API.

    Satisfies the Captioner protocol. Uses disk cache to avoid redundant
    API calls. Handles ffmpeg clip extraction and cleanup even on error.

    Note: Class retains its original name for backwards compatibility with
    CLI and ingestion imports, even though it now uses OpenRouter exclusively.
    """

    def __init__(self, settings: Settings) -> None:
        self._openrouter_key = settings.openrouter_api_key
        self._model = settings.captioner_model  # e.g. "google/gemini-2.5-flash"
        # Ensure model has provider prefix for OpenRouter
        if "/" not in self._model:
            self._model = f"google/{self._model}"
        self._cache_dir = settings.caption_cache_dir
        self._ffmpeg_path = settings.ffmpeg_path

    def caption(
        self,
        video_path: str,
        start: float,
        end: float,
        chunk_index: int | None = None,
    ) -> dict:
        """Extract a visual caption for a video chunk.

        Args:
            video_path: Path to the source video file.
            start: Start time in seconds.
            end: End time in seconds.
            chunk_index: Explicit chunk index for the cache key. When omitted,
                falls back to int(start) for protocol-only callers. Callers with
                access to the real chunk_index (e.g. CLI) should always pass it
                so cache keys stay stable if chunk timing changes.

        Returns:
            Dict with keys: "caption" (str), "cached" (bool).
        """
        video_id = Path(video_path).stem.replace("_720p", "")
        if chunk_index is None:
            chunk_index = int(start)

        # Cache-first: return cached value without any API call
        cached_text = self._load_cache(video_id, chunk_index)
        if cached_text is not None:
            return {"caption": cached_text, "cached": True}

        if not self._openrouter_key:
            raise QuotaExhaustedError(
                "No OpenRouter API key configured -- cannot caption"
            )

        # Extract clip to temp file
        clip_path = self._extract_clip(video_path, start, end)
        try:
            caption_text = self._call_openrouter(clip_path)
        finally:
            os.unlink(clip_path)

        if not caption_text:
            raise RuntimeError(f"API returned empty/null caption for chunk {chunk_index}")

        self._save_cache(video_id, chunk_index, caption_text)
        return {"caption": caption_text, "cached": False}

    def _call_openrouter(self, clip_path: str) -> str:
        """Caption via OpenRouter with base64-encoded video clip."""
        video_bytes = Path(clip_path).read_bytes()
        video_b64 = base64.b64encode(video_bytes).decode()
        data_url = f"data:video/mp4;base64,{video_b64}"

        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": data_url},
                            },
                            {
                                "type": "text",
                                "text": CAPTION_USER_PROMPT,
                            },
                        ],
                    },
                ],
                "temperature": 0.0,
            },
            timeout=90.0,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            error_msg = data["error"]
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
            if "429" in str(error_msg) or "quota" in str(error_msg).lower():
                raise QuotaExhaustedError(f"OpenRouter quota exhausted: {error_msg}")
            raise RuntimeError(f"OpenRouter error: {error_msg}")
        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"OpenRouter response missing choices: {list(data.keys())}")

        return data["choices"][0]["message"]["content"]

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
                "-i", video_path,
                "-ss", str(start),
                "-t", str(end - start),
                "-an",
                "-crf", "28",
                tmp_path,
            ],
            check=True,
            capture_output=True,
        )
        return tmp_path

    def is_cached(self, video_id: str, chunk_index: int) -> bool:
        """Check if a valid (non-null) cache entry exists for a chunk."""
        return self._load_cache(video_id, chunk_index) is not None

    def _cache_path(self, video_id: str, chunk_index: int) -> Path:
        """Return the cache file path for a given video/chunk."""
        return Path(self._cache_dir) / video_id / f"{chunk_index}.json"

    def _load_cache(self, video_id: str, chunk_index: int) -> Optional[str]:
        """Load a cached caption if it exists.

        Returns:
            Cached caption string, or None if no cache entry or if the cached
            value is null/empty (invalid cache entry).
        """
        path = self._cache_path(video_id, chunk_index)
        if path.exists():
            data = json.loads(path.read_text())
            caption = data.get("caption")
            if not caption:
                path.unlink(missing_ok=True)
                return None
            return caption
        return None

    def _save_cache(self, video_id: str, chunk_index: int, caption: str) -> None:
        """Write a caption to the disk cache."""
        path = self._cache_path(video_id, chunk_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "caption": caption,
            "model": self._model,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(cache_data))
