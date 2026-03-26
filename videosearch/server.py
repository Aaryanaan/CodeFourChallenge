"""Minimal FastAPI server wrapping HybridRetriever (INT-06).

Single endpoint: POST /search
Swagger UI: /docs
"""

from fastapi import FastAPI
from pydantic import BaseModel

from videosearch.classifier import GeminiQueryClassifier
from videosearch.config import Settings
from videosearch.hybrid_retriever import HybridRetriever
from videosearch.reranker import ClaudeReranker


app = FastAPI(title="VideoSearch API", description="Search body-worn camera footage")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    score: float
    transcript_snippet: str
    visual_caption: str
    reasoning: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Search video footage with a natural language query."""
    settings = Settings()
    classifier = GeminiQueryClassifier(settings)
    reranker = ClaudeReranker(settings)
    retriever = HybridRetriever(settings, classifier=classifier, reranker=reranker)
    raw = retriever.retrieve(req.query, top_k=req.top_k)
    results = [
        SearchResult(
            video_id=r["video_id"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            score=r.get("rrf_score", 0),
            transcript_snippet=r.get("transcript_snippet", r.get("combined_text", ""))[:200],
            visual_caption=r.get("visual_caption", "")[:200],
            reasoning=r.get("reasoning", ""),
        )
        for r in raw
    ]
    return SearchResponse(query=req.query, results=results)
