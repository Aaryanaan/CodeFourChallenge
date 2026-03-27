"""Tests for GeminiQueryClassifier (OpenRouter backend).

Covers query classification into 5 types with locked weight policies,
disk caching, and protocol conformance.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from videosearch.config import Settings
from videosearch.classifier import GeminiQueryClassifier, WEIGHT_POLICY
from videosearch.protocols import Classifier


@pytest.fixture
def settings(tmp_path):
    """Create Settings with fake API key and temp cache dir."""
    return Settings(
        openrouter_api_key="fake-openrouter-key",
        classifier_cache_dir=tmp_path / "classifier",
        _env_file=None,
    )


def _mock_openrouter_response(query_type: str) -> MagicMock:
    """Create a mock httpx response returning query_type JSON."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"query_type": query_type})}}]
    }
    return resp


class TestClassifyTranscript:
    def test_classify_transcript(self, settings):
        """Query about Miranda rights -> transcript type with BM25-heavy weights."""
        with patch("videosearch.classifier.httpx.post", return_value=_mock_openrouter_response("transcript")):
            classifier = GeminiQueryClassifier(settings)
            result = classifier.classify(
                "Find all interactions where an officer reads Miranda rights"
            )
        assert result["query_type"] == "transcript"
        assert result["weights"] == {"vector": 0.3, "bm25": 2.5, "filter": 0.0}


class TestClassifyVisual:
    def test_classify_visual(self, settings):
        """Query about red shirt -> visual type with vector-heavy weights."""
        with patch("videosearch.classifier.httpx.post", return_value=_mock_openrouter_response("visual")):
            classifier = GeminiQueryClassifier(settings)
            result = classifier.classify(
                "Locate all footage containing a person in a red shirt"
            )
        assert result["query_type"] == "visual"
        assert result["weights"] == {"vector": 2.5, "bm25": 0.3, "filter": 0.0}


class TestClassifyAudio:
    def test_classify_audio(self, settings):
        """Query about raised voice -> audio type with filter-heavy weights."""
        with patch("videosearch.classifier.httpx.post", return_value=_mock_openrouter_response("audio")):
            classifier = GeminiQueryClassifier(settings)
            result = classifier.classify(
                "Find every moment where someone raises their voice"
            )
        assert result["query_type"] == "audio"
        assert result["weights"] == {"vector": 0.3, "bm25": 0.5, "filter": 2.5}


class TestClassifyTemporal:
    def test_classify_temporal(self, settings):
        """Query about night -> temporal type with vector-heavy weights."""
        with patch("videosearch.classifier.httpx.post", return_value=_mock_openrouter_response("temporal")):
            classifier = GeminiQueryClassifier(settings)
            result = classifier.classify(
                "Find all instances of a vehicle being pulled over at night"
            )
        assert result["query_type"] == "temporal"
        assert result["weights"] == {"vector": 2.0, "bm25": 0.3, "filter": 0.0}


class TestClassifyMixed:
    def test_classify_mixed(self, settings):
        """Mixed modality query -> mixed type with balanced weights."""
        with patch("videosearch.classifier.httpx.post", return_value=_mock_openrouter_response("mixed")):
            classifier = GeminiQueryClassifier(settings)
            result = classifier.classify(
                "Find moments where someone in a red shirt is yelling"
            )
        assert result["query_type"] == "mixed"
        assert result["weights"] == {"vector": 1.0, "bm25": 1.0, "filter": 0.5}


class TestWeightDominance:
    def test_weight_dominance(self, settings):
        """Verify dominant weights: transcript bm25>=2.0, audio filter>=2.0,
        visual vector>=2.0, temporal vector>=2.0."""
        cases = [
            ("transcript", "bm25", 2.0),
            ("audio", "filter", 2.0),
            ("visual", "vector", 2.0),
            ("temporal", "vector", 2.0),
        ]

        for query_type, dominant_key, min_val in cases:
            with patch("videosearch.classifier.httpx.post", return_value=_mock_openrouter_response(query_type)):
                classifier = GeminiQueryClassifier(settings)
                result = classifier.classify(f"test query for {query_type}")
            assert result["weights"][dominant_key] >= min_val, (
                f"{query_type}: {dominant_key} should be >= {min_val}, "
                f"got {result['weights'][dominant_key]}"
            )


class TestCacheHit:
    def test_cache_hit(self, settings):
        """Second classify with same query uses cache, no API call."""
        with patch("videosearch.classifier.httpx.post", return_value=_mock_openrouter_response("transcript")) as mock_post:
            classifier = GeminiQueryClassifier(settings)

            query = "Find all interactions where an officer reads Miranda rights"
            result1 = classifier.classify(query)
            result2 = classifier.classify(query)

            assert result1 == result2
            # httpx.post should only be called once
            assert mock_post.call_count == 1


class TestCacheKeyFormat:
    def test_cache_key_format(self, settings):
        """Cache key is sha256(query.lower().strip())[:16] hex chars."""
        import hashlib

        with patch("videosearch.classifier.httpx.post", return_value=_mock_openrouter_response("visual")):
            classifier = GeminiQueryClassifier(settings)
            query = "  Test Query  "
            classifier.classify(query)

        expected_key = hashlib.sha256(
            query.lower().strip().encode()
        ).hexdigest()[:16]
        cache_file = settings.classifier_cache_dir / f"{expected_key}.json"
        assert cache_file.exists()


class TestClassifierFallback:
    def test_classifier_falls_back_to_mixed_on_api_failure(self, settings):
        """API failures should not abort search; classifier falls back to mixed."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = RuntimeError("boom")

        with patch("videosearch.classifier.httpx.post", return_value=mock_response):
            classifier = GeminiQueryClassifier(settings)
            result = classifier.classify("Find something")

        assert result["query_type"] == "mixed"
        assert result["weights"] == WEIGHT_POLICY["mixed"]


class TestClassifierProtocol:
    def test_classifier_protocol(self, settings):
        """GeminiQueryClassifier satisfies Classifier protocol."""
        classifier = GeminiQueryClassifier(settings)
        assert isinstance(classifier, Classifier)
