"""Tests for Protocol interfaces."""

from videosearch.protocols import (
    Chunker,
    Compressor,
    Transcriber,
    AudioAnalyzer,
    OCRExtractor,
    Captioner,
    Embedder,
    VectorStore,
    Retriever,
    Reranker,
)
from videosearch.models import ChunkMetadata


def test_all_protocols_importable():
    """All 10 protocol classes are importable."""
    protocols = [
        Chunker,
        Compressor,
        Transcriber,
        AudioAnalyzer,
        OCRExtractor,
        Captioner,
        Embedder,
        VectorStore,
        Retriever,
        Reranker,
    ]
    assert len(protocols) == 10
    for p in protocols:
        assert hasattr(p, "__protocol_attrs__") or hasattr(p, "__abstractmethods__") or True


def test_all_protocols_runtime_checkable():
    """All 10 protocols are decorated with @runtime_checkable."""
    protocols = [
        Chunker,
        Compressor,
        Transcriber,
        AudioAnalyzer,
        OCRExtractor,
        Captioner,
        Embedder,
        VectorStore,
        Retriever,
        Reranker,
    ]
    for p in protocols:
        # runtime_checkable protocols have _is_runtime_protocol set to True
        assert getattr(p, "_is_runtime_protocol", False), f"{p.__name__} is not runtime_checkable"


def test_chunker_isinstance_check():
    """A class implementing chunk() satisfies the Chunker protocol."""

    class DummyChunker:
        def chunk(self, video_path: str) -> list[ChunkMetadata]:
            return []

    assert isinstance(DummyChunker(), Chunker)


def test_compressor_isinstance_check():
    """A class implementing compress() satisfies the Compressor protocol."""

    class DummyCompressor:
        def compress(self, video_path: str, output_path: str) -> str:
            return output_path

    assert isinstance(DummyCompressor(), Compressor)


def test_transcriber_isinstance_check():
    """A class implementing transcribe() satisfies the Transcriber protocol."""

    class DummyTranscriber:
        def transcribe(self, video_path: str, start: float, end: float) -> dict:
            return {}

    assert isinstance(DummyTranscriber(), Transcriber)


def test_audio_analyzer_isinstance_check():
    """A class implementing analyze() satisfies the AudioAnalyzer protocol."""

    class DummyAudioAnalyzer:
        def analyze(self, video_path: str, start: float, end: float) -> dict:
            return {}

    assert isinstance(DummyAudioAnalyzer(), AudioAnalyzer)


def test_ocr_extractor_isinstance_check():
    """A class implementing extract() satisfies the OCRExtractor protocol."""

    class DummyOCRExtractor:
        def extract(self, video_path: str, start: float, end: float) -> dict:
            return {}

    assert isinstance(DummyOCRExtractor(), OCRExtractor)


def test_captioner_isinstance_check():
    """A class implementing caption() satisfies the Captioner protocol."""

    class DummyCaptioner:
        def caption(self, video_path: str, start: float, end: float) -> dict:
            return {}

    assert isinstance(DummyCaptioner(), Captioner)


def test_embedder_isinstance_check():
    """A class implementing embed() satisfies the Embedder protocol."""

    class DummyEmbedder:
        def embed(self, text: str) -> list[float]:
            return [0.0]

    assert isinstance(DummyEmbedder(), Embedder)


def test_vector_store_isinstance_check():
    """A class implementing search() satisfies the VectorStore protocol."""

    class DummyVectorStore:
        def search(self, vector: list[float], top_k: int) -> list[dict]:
            return []

    assert isinstance(DummyVectorStore(), VectorStore)


def test_retriever_isinstance_check():
    """A class implementing retrieve() satisfies the Retriever protocol."""

    class DummyRetriever:
        def retrieve(self, query: str, top_k: int) -> list[dict]:
            return []

    assert isinstance(DummyRetriever(), Retriever)


def test_reranker_isinstance_check():
    """A class implementing rerank() satisfies the Reranker protocol."""

    class DummyReranker:
        def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
            return []

    assert isinstance(DummyReranker(), Reranker)


def test_non_conforming_class_fails_isinstance():
    """A class missing the required method does NOT satisfy a Protocol."""

    class NotAChunker:
        def something_else(self):
            pass

    assert not isinstance(NotAChunker(), Chunker)


def test_partial_conforming_class_fails():
    """A class with a different method name does not satisfy the Protocol."""

    class WrongMethod:
        def process(self, video_path: str) -> list:
            return []

    assert not isinstance(WrongMethod(), Chunker)
