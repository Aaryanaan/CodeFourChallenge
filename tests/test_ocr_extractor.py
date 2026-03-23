"""Tests for PaddleOCRExtractor with mocked OCR engines."""

from unittest.mock import MagicMock, patch, PropertyMock
import types

import numpy as np
import pytest

from videosearch.ocr_extractor import (
    PaddleOCRExtractor,
    is_duplicate,
    deduplicate_ocr,
    sample_frames,
)
from videosearch.protocols import OCRExtractor


class TestProtocolConformance:
    """PaddleOCRExtractor satisfies OCRExtractor protocol."""

    def test_satisfies_protocol(self):
        extractor = PaddleOCRExtractor()
        assert isinstance(extractor, OCRExtractor)


class TestIsDuplicate:
    """Fuzzy string matching for OCR deduplication."""

    def test_identical_strings(self):
        assert is_duplicate("ABC123", "ABC123") is True

    def test_case_insensitive(self):
        assert is_duplicate("ABC123", "abc123") is True

    def test_similar_strings_above_threshold(self):
        # "ABC123" vs "ABC12B" -- 5/6 chars match = 0.833, but SequenceMatcher
        # may give higher ratio due to matching blocks
        assert is_duplicate("ABC123", "ABC12B", threshold=0.8) is True

    def test_different_strings_below_threshold(self):
        assert is_duplicate("ABC123", "XYZ789") is False

    def test_empty_strings(self):
        assert is_duplicate("", "") is True


class TestDeduplicateOcr:
    """Merge OCR results from multiple frames."""

    def test_deduplication_merges_same_text(self):
        frame_results = [
            {"text": "ABC123", "confidence": 0.9, "timestamp": 2.0,
             "bbox": [[0, 0], [100, 0], [100, 30], [0, 30]]},
            {"text": "ABC123", "confidence": 0.95, "timestamp": 4.0,
             "bbox": [[0, 0], [100, 0], [100, 30], [0, 30]]},
            {"text": "ABC123", "confidence": 0.85, "timestamp": 6.0,
             "bbox": [[0, 0], [100, 0], [100, 30], [0, 30]]},
        ]
        results = deduplicate_ocr(frame_results)
        assert len(results) == 1
        assert results[0]["first_seen"] == 2.0
        assert results[0]["last_seen"] == 6.0
        # Should keep highest confidence text/bbox
        assert results[0]["confidence"] == 0.95

    def test_deduplication_keeps_distinct_texts(self):
        frame_results = [
            {"text": "ABC123", "confidence": 0.9, "timestamp": 2.0,
             "bbox": [[0, 0], [100, 0], [100, 30], [0, 30]]},
            {"text": "XYZ789", "confidence": 0.88, "timestamp": 4.0,
             "bbox": [[200, 0], [300, 0], [300, 30], [200, 30]]},
        ]
        results = deduplicate_ocr(frame_results)
        assert len(results) == 2

    def test_fuzzy_deduplication(self):
        """Similar strings (e.g., OCR misread) should be merged when similarity >= 0.85."""
        frame_results = [
            {"text": "ABC1234", "confidence": 0.9, "timestamp": 2.0,
             "bbox": [[0, 0], [100, 0], [100, 30], [0, 30]]},
            {"text": "ABC12B4", "confidence": 0.75, "timestamp": 4.0,
             "bbox": [[0, 0], [100, 0], [100, 30], [0, 30]]},
        ]
        results = deduplicate_ocr(frame_results)
        assert len(results) == 1
        # Higher confidence version kept
        assert results[0]["text"] == "ABC1234"
        assert results[0]["confidence"] == 0.9

    def test_empty_input(self):
        assert deduplicate_ocr([]) == []


class TestPaddleOCRExtractorExtract:
    """PaddleOCRExtractor.extract() with mocked OCR engine and video capture."""

    def _make_mock_capture(self, frame_count=3, width=640, height=480):
        """Create a mock cv2.VideoCapture that returns synthetic frames."""
        cap = MagicMock()
        self._read_count = 0
        self._max_reads = frame_count

        def mock_read():
            if self._read_count < self._max_reads:
                self._read_count += 1
                frame = np.ones((height, width, 3), dtype=np.uint8) * 128
                return True, frame
            return False, None

        cap.read = mock_read
        cap.set = MagicMock()
        cap.get = MagicMock(side_effect=lambda prop: 0.0)
        cap.release = MagicMock()
        return cap

    def _make_paddle_mock(self, texts, scores, polys):
        """Create a mock PaddleOCR engine that returns given results on each predict() call."""
        mock_engine = MagicMock()

        def make_result():
            mock_result = MagicMock()
            mock_result.rec_texts = texts
            mock_result.rec_scores = scores
            mock_result.rec_polys = polys
            return mock_result

        mock_engine.predict = MagicMock(side_effect=lambda frame: iter([make_result()]))
        return mock_engine

    def test_extract_returns_dict_with_results_key(self):
        extractor = PaddleOCRExtractor(confidence_threshold=0.7, frame_interval=2.0)
        mock_cap = self._make_mock_capture(frame_count=2)

        mock_engine = self._make_paddle_mock(
            ["HELLO"], [0.95],
            [np.array([[0, 0], [100, 0], [100, 30], [0, 30]])]
        )

        extractor._engine = mock_engine
        extractor._backend = "paddleocr"

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with patch("os.path.exists", return_value=True):
                result = extractor.extract("/fake/video.mp4", 0.0, 6.0)

        assert "results" in result
        assert isinstance(result["results"], list)

    def test_extract_result_has_required_keys(self):
        extractor = PaddleOCRExtractor(confidence_threshold=0.7, frame_interval=2.0)
        mock_cap = self._make_mock_capture(frame_count=1)

        mock_engine = self._make_paddle_mock(
            ["PLATE42"], [0.92],
            [np.array([[10, 10], [110, 10], [110, 40], [10, 40]])]
        )

        extractor._engine = mock_engine
        extractor._backend = "paddleocr"

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with patch("os.path.exists", return_value=True):
                result = extractor.extract("/fake/video.mp4", 0.0, 2.0)

        assert len(result["results"]) >= 1
        ocr_item = result["results"][0]
        assert "text" in ocr_item
        assert "confidence" in ocr_item
        assert "first_seen" in ocr_item
        assert "last_seen" in ocr_item
        assert "bbox" in ocr_item

    def test_confidence_filter(self):
        """Results below confidence threshold are filtered out."""
        extractor = PaddleOCRExtractor(confidence_threshold=0.7, frame_interval=2.0)
        mock_cap = self._make_mock_capture(frame_count=1)

        mock_engine = self._make_paddle_mock(
            ["CLEAR", "FUZZY"], [0.95, 0.3],
            [
                np.array([[0, 0], [100, 0], [100, 30], [0, 30]]),
                np.array([[200, 0], [300, 0], [300, 30], [200, 30]]),
            ]
        )

        extractor._engine = mock_engine
        extractor._backend = "paddleocr"

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with patch("os.path.exists", return_value=True):
                result = extractor.extract("/fake/video.mp4", 0.0, 2.0)

        texts = [r["text"] for r in result["results"]]
        assert "CLEAR" in texts
        assert "FUZZY" not in texts

    def test_deduplication_across_frames(self):
        """Same text in multiple frames is merged with first/last seen."""
        extractor = PaddleOCRExtractor(confidence_threshold=0.7, frame_interval=2.0)
        mock_cap = self._make_mock_capture(frame_count=3)

        mock_engine = self._make_paddle_mock(
            ["STOP"], [0.9],
            [np.array([[0, 0], [80, 0], [80, 25], [0, 25]])]
        )

        extractor._engine = mock_engine
        extractor._backend = "paddleocr"

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with patch("os.path.exists", return_value=True):
                result = extractor.extract("/fake/video.mp4", 0.0, 6.0)

        # Should be deduplicated to one entry
        assert len(result["results"]) == 1
        assert result["results"][0]["text"] == "STOP"

    def test_empty_results_for_no_text(self):
        """Frames with no detected text return empty results list."""
        extractor = PaddleOCRExtractor(confidence_threshold=0.7, frame_interval=2.0)
        mock_cap = self._make_mock_capture(frame_count=2)

        mock_engine = self._make_paddle_mock([], [], [])

        extractor._engine = mock_engine
        extractor._backend = "paddleocr"

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with patch("os.path.exists", return_value=True):
                result = extractor.extract("/fake/video.mp4", 0.0, 4.0)

        assert result["results"] == []

    def test_easyocr_fallback(self):
        """When PaddleOCR import fails, EasyOCR is used as fallback."""
        extractor = PaddleOCRExtractor(confidence_threshold=0.7, frame_interval=2.0)

        # Simulate PaddleOCR import failure and EasyOCR fallback
        mock_easyocr_engine = MagicMock()
        mock_easyocr_engine.readtext = MagicMock(return_value=[
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "EASYTEXT", 0.88),
        ])

        def mock_init_engine(self_inner):
            self_inner._engine = mock_easyocr_engine
            self_inner._backend = "easyocr"

        extractor._init_engine = types.MethodType(mock_init_engine, extractor)

        mock_cap = self._make_mock_capture(frame_count=1)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with patch("os.path.exists", return_value=True):
                result = extractor.extract("/fake/video.mp4", 0.0, 2.0)

        assert result["backend"] == "easyocr"
        assert len(result["results"]) == 1
        assert result["results"][0]["text"] == "EASYTEXT"

    def test_frame_count_in_result(self):
        """extract() returns frame_count indicating how many frames were sampled."""
        extractor = PaddleOCRExtractor(confidence_threshold=0.7, frame_interval=2.0)
        mock_cap = self._make_mock_capture(frame_count=3)

        mock_engine = self._make_paddle_mock([], [], [])

        extractor._engine = mock_engine
        extractor._backend = "paddleocr"

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with patch("os.path.exists", return_value=True):
                result = extractor.extract("/fake/video.mp4", 0.0, 6.0)

        assert "frame_count" in result
        assert result["frame_count"] == 3

    def test_file_not_found_raises(self):
        """extract() raises FileNotFoundError for nonexistent video."""
        extractor = PaddleOCRExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract("/nonexistent/video.mp4", 0.0, 5.0)


class TestEasyOCRBackend:
    """Test EasyOCR backend integration path."""

    def test_easyocr_init_fallback(self):
        """When PaddleOCR is unavailable, _init_engine falls back to EasyOCR."""
        extractor = PaddleOCRExtractor()

        mock_reader = MagicMock()

        with patch.dict("sys.modules", {"paddleocr": None}):
            with patch("builtins.__import__", side_effect=_import_side_effect(mock_reader)):
                extractor._init_engine()

        assert extractor._backend == "easyocr"
        assert extractor._engine is mock_reader


def _import_side_effect(mock_reader):
    """Side effect for __import__ that fails paddleocr but returns easyocr mock."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def side_effect(name, *args, **kwargs):
        if name == "paddleocr":
            raise ImportError("No module named 'paddleocr'")
        if name == "easyocr":
            mock_module = MagicMock()
            mock_module.Reader = MagicMock(return_value=mock_reader)
            return mock_module
        return real_import(name, *args, **kwargs)

    return side_effect
