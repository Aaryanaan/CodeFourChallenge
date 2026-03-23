"""OCR extraction from video frames using PaddleOCR with EasyOCR fallback.

Samples frames at regular intervals from a video segment, runs OCR,
filters by confidence threshold, and deduplicates text across frames
using fuzzy string matching.
"""

import os
from difflib import SequenceMatcher

import cv2


def sample_frames(video_path: str, start: float, end: float, interval: float = 2.0):
    """Yield (timestamp, frame) tuples at regular intervals from video segment."""
    cap = cv2.VideoCapture(video_path)
    try:
        t = start
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break
            actual_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield actual_t, frame
            t += interval
    finally:
        cap.release()


def is_duplicate(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """Check if two OCR reads are likely the same text using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() >= threshold


def deduplicate_ocr(frame_results: list[dict]) -> list[dict]:
    """Merge OCR results from multiple frames, keeping first/last timestamps.

    If a duplicate is found with higher confidence, the text, confidence,
    and bbox are updated to the higher-confidence version.
    """
    unique: list[dict] = []
    for result in frame_results:
        merged = False
        for existing in unique:
            if is_duplicate(result["text"], existing["text"]):
                existing["last_seen"] = result["timestamp"]
                if result["confidence"] > existing["confidence"]:
                    existing["text"] = result["text"]
                    existing["confidence"] = result["confidence"]
                    existing["bbox"] = result["bbox"]
                merged = True
                break
        if not merged:
            unique.append({
                "text": result["text"],
                "confidence": result["confidence"],
                "first_seen": result["timestamp"],
                "last_seen": result["timestamp"],
                "bbox": result["bbox"],
            })
    return unique


class PaddleOCRExtractor:
    """Extract text from video frames using OCR. Satisfies OCRExtractor protocol.

    Uses PaddleOCR 3.x as the primary engine, with EasyOCR as fallback.
    Samples frames at configurable intervals, filters results by confidence
    threshold, and deduplicates text across frames using fuzzy matching.
    """

    def __init__(self, confidence_threshold: float = 0.7, frame_interval: float = 2.0):
        self.confidence_threshold = confidence_threshold
        self.frame_interval = frame_interval
        self._engine = None
        self._backend: str | None = None

    def _init_engine(self):
        """Lazy-initialize OCR engine. Try PaddleOCR first, fall back to EasyOCR."""
        if self._engine is not None:
            return
        try:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                lang="en",
            )
            self._backend = "paddleocr"
        except Exception:
            import easyocr

            self._engine = easyocr.Reader(["en"], gpu=False)
            self._backend = "easyocr"

    def _run_ocr(self, frame) -> list[dict]:
        """Run OCR on a single frame. Returns list of {text, confidence, bbox}."""
        results = []

        if self._backend == "paddleocr":
            result = next(self._engine.predict(frame))
            for text, score, poly in zip(
                result.rec_texts, result.rec_scores, result.rec_polys
            ):
                if score >= self.confidence_threshold:
                    # Normalize poly to list[list[float]]
                    bbox = [[float(p[0]), float(p[1])] for p in poly]
                    results.append({
                        "text": text,
                        "confidence": float(score),
                        "bbox": bbox,
                    })
        elif self._backend == "easyocr":
            detections = self._engine.readtext(frame)
            for bbox, text, confidence in detections:
                if confidence >= self.confidence_threshold:
                    # Normalize bbox to list[list[float]]
                    normalized_bbox = [[float(p[0]), float(p[1])] for p in bbox]
                    results.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": normalized_bbox,
                    })

        return results

    def extract(self, video_path: str, start: float, end: float) -> dict:
        """Extract text from sampled video frames. Returns OCR results dict.

        Returns:
            dict with keys:
                - results: list of {text, confidence, first_seen, last_seen, bbox}
                - frame_count: number of frames sampled
                - backend: "paddleocr" or "easyocr"
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._init_engine()

        all_frame_results: list[dict] = []
        frame_count = 0

        for timestamp, frame in sample_frames(
            video_path, start, end, self.frame_interval
        ):
            frame_count += 1
            ocr_hits = self._run_ocr(frame)
            for hit in ocr_hits:
                hit["timestamp"] = timestamp
                all_frame_results.append(hit)

        deduplicated = deduplicate_ocr(all_frame_results)

        return {
            "results": deduplicated,
            "frame_count": frame_count,
            "backend": self._backend,
        }
