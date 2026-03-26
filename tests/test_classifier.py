"""Tests for GeminiQueryClassifier.

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
        google_api_key="fake-key",
        classifier_cache_dir=tmp_path / "classifier",
        _env_file=None,
    )


@pytest.fixture
def mock_genai():
    """Patch genai.Client so no real API calls are made."""
    with patch("videosearch.classifier.genai.Client") as mock_client_cls:
        yield mock_client_cls


def _make_response(query_type: str, weights: dict) -> MagicMock:
    """Create a mock generate_content response."""
    resp = MagicMock()
    resp.text = json.dumps({"query_type": query_type, "weights": weights})
    return resp


class TestClassifyTranscript:
    def test_classify_transcript(self, settings, mock_genai):
        """Query about Miranda rights -> transcript type with BM25-heavy weights."""
        mock_client = mock_genai.return_value
        mock_client.models.generate_content.return_value = _make_response(
            "transcript", {"vector": 0.5, "bm25": 1.5, "filter": 0.2}
        )
        classifier = GeminiQueryClassifier(settings)
        result = classifier.classify(
            "Find all interactions where an officer reads Miranda rights"
        )
        assert result["query_type"] == "transcript"
        assert result["weights"] == {"vector": 0.3, "bm25": 2.5, "filter": 0.0}


class TestClassifyVisual:
    def test_classify_visual(self, settings, mock_genai):
        """Query about red shirt -> visual type with vector-heavy weights."""
        mock_client = mock_genai.return_value
        mock_client.models.generate_content.return_value = _make_response(
            "visual", {"vector": 1.0, "bm25": 1.0, "filter": 0.0}
        )
        classifier = GeminiQueryClassifier(settings)
        result = classifier.classify(
            "Locate all footage containing a person in a red shirt"
        )
        assert result["query_type"] == "visual"
        assert result["weights"] == {"vector": 2.5, "bm25": 0.3, "filter": 0.0}


class TestClassifyAudio:
    def test_classify_audio(self, settings, mock_genai):
        """Query about raised voice -> audio type with filter-heavy weights."""
        mock_client = mock_genai.return_value
        mock_client.models.generate_content.return_value = _make_response(
            "audio", {"vector": 0.5, "bm25": 0.5, "filter": 1.0}
        )
        classifier = GeminiQueryClassifier(settings)
        result = classifier.classify(
            "Find every moment where someone raises their voice"
        )
        assert result["query_type"] == "audio"
        assert result["weights"] == {"vector": 0.3, "bm25": 0.5, "filter": 2.5}


class TestClassifyTemporal:
    def test_classify_temporal(self, settings, mock_genai):
        """Query about night -> temporal type with vector-heavy weights."""
        mock_client = mock_genai.return_value
        mock_client.models.generate_content.return_value = _make_response(
            "temporal", {"vector": 1.0, "bm25": 0.5, "filter": 0.0}
        )
        classifier = GeminiQueryClassifier(settings)
        result = classifier.classify(
            "Find all instances of a vehicle being pulled over at night"
        )
        assert result["query_type"] == "temporal"
        assert result["weights"] == {"vector": 2.0, "bm25": 0.3, "filter": 0.0}


class TestClassifyMixed:
    def test_classify_mixed(self, settings, mock_genai):
        """Mixed modality query -> mixed type with balanced weights."""
        mock_client = mock_genai.return_value
        mock_client.models.generate_content.return_value = _make_response(
            "mixed", {"vector": 0.8, "bm25": 0.8, "filter": 0.4}
        )
        classifier = GeminiQueryClassifier(settings)
        result = classifier.classify(
            "Find moments where someone in a red shirt is yelling"
        )
        assert result["query_type"] == "mixed"
        assert result["weights"] == {"vector": 1.0, "bm25": 1.0, "filter": 0.5}


class TestWeightDominance:
    def test_weight_dominance(self, settings, mock_genai):
        """Verify dominant weights: transcript bm25>=2.0, audio filter>=2.0,
        visual vector>=2.0, temporal vector>=2.0."""
        mock_client = mock_genai.return_value

        cases = [
            ("transcript", {"vector": 0.3, "bm25": 2.5, "filter": 0.0}, "bm25", 2.0),
            ("audio", {"vector": 0.3, "bm25": 0.5, "filter": 2.5}, "filter", 2.0),
            ("visual", {"vector": 2.5, "bm25": 0.3, "filter": 0.0}, "vector", 2.0),
            ("temporal", {"vector": 2.0, "bm25": 0.3, "filter": 0.0}, "vector", 2.0),
        ]

        for query_type, expected_weights, dominant_key, min_val in cases:
            mock_client.models.generate_content.return_value = _make_response(
                query_type, {"vector": 1.0, "bm25": 1.0, "filter": 1.0}
            )
            classifier = GeminiQueryClassifier(settings)
            result = classifier.classify(f"test query for {query_type}")
            assert result["weights"][dominant_key] >= min_val, (
                f"{query_type}: {dominant_key} should be >= {min_val}, "
                f"got {result['weights'][dominant_key]}"
            )


class TestCacheHit:
    def test_cache_hit(self, settings, mock_genai):
        """Second classify with same query uses cache, no API call."""
        mock_client = mock_genai.return_value
        mock_client.models.generate_content.return_value = _make_response(
            "transcript", {"vector": 0.3, "bm25": 2.5, "filter": 0.0}
        )
        classifier = GeminiQueryClassifier(settings)

        query = "Find all interactions where an officer reads Miranda rights"
        result1 = classifier.classify(query)
        result2 = classifier.classify(query)

        assert result1 == result2
        # generate_content should only be called once
        assert mock_client.models.generate_content.call_count == 1


class TestCacheKeyFormat:
    def test_cache_key_format(self, settings, mock_genai):
        """Cache key is sha256(query.lower().strip())[:16] hex chars."""
        import hashlib

        mock_client = mock_genai.return_value
        mock_client.models.generate_content.return_value = _make_response(
            "visual", {"vector": 2.5, "bm25": 0.3, "filter": 0.0}
        )
        classifier = GeminiQueryClassifier(settings)
        query = "  Test Query  "
        classifier.classify(query)

        expected_key = hashlib.sha256(
            query.lower().strip().encode()
        ).hexdigest()[:16]
        cache_file = settings.classifier_cache_dir / f"{expected_key}.json"
        assert cache_file.exists()


class TestClassifierFallback:
    def test_classifier_falls_back_to_mixed_on_api_failure(self, settings, mock_genai):
        """API failures should not abort search; classifier falls back to mixed."""
        mock_client = mock_genai.return_value
        mock_client.models.generate_content.side_effect = RuntimeError("boom")

        classifier = GeminiQueryClassifier(settings)
        result = classifier.classify("Find something")

        assert result["query_type"] == "mixed"
        assert result["weights"] == WEIGHT_POLICY["mixed"]


class TestClassifierProtocol:
    def test_classifier_protocol(self, settings, mock_genai):
        """GeminiQueryClassifier satisfies Classifier protocol."""
        classifier = GeminiQueryClassifier(settings)
        assert isinstance(classifier, Classifier)
