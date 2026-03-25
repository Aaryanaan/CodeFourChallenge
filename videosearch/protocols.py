"""Protocol interfaces for all pipeline components.

Each protocol defines a single method for its component. Implementations
satisfy the protocol structurally -- no inheritance required. All protocols
are @runtime_checkable for isinstance() assertions.
"""

from typing import Protocol, runtime_checkable

from videosearch.models import ChunkMetadata


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, video_path: str) -> list[ChunkMetadata]: ...


@runtime_checkable
class Compressor(Protocol):
    def compress(self, video_path: str, output_path: str) -> str: ...


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, video_path: str, start: float, end: float) -> dict: ...


@runtime_checkable
class AudioAnalyzer(Protocol):
    def analyze(self, video_path: str, start: float, end: float) -> dict: ...


@runtime_checkable
class OCRExtractor(Protocol):
    def extract(self, video_path: str, start: float, end: float) -> dict: ...


@runtime_checkable
class Captioner(Protocol):
    def caption(self, video_path: str, start: float, end: float) -> dict: ...


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    def search(self, vector: list[float], top_k: int) -> list[dict]: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[dict]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]: ...


@runtime_checkable
class Classifier(Protocol):
    def classify(self, query: str) -> dict: ...
