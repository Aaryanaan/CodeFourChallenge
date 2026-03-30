# VideoSearch

Natural language search for body-worn camera footage. Type a query in plain English; get back ranked video segments with timestamps, transcript excerpts, visual captions, and reranker reasoning.

## Requirements

- Python 3.11+
- `ffmpeg` (required for video compression, chunking, and audio extraction)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `apt install ffmpeg libgl1 libglib2.0-0`
- API keys (see [Environment Variables](#environment-variables))

## Installation

```bash
pip install -e ".[dev]"
```

## Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-google-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

`GOOGLE_API_KEY` — used by the Gemini embedder at index and query time. A local `all-MiniLM-L6-v2` fallback exists, but the pre-built index uses 768-dim Gemini embeddings — omitting this key will cause a dimension mismatch on vector search.

`OPENROUTER_API_KEY` — used by the query classifier (Gemini Flash) and reranker (Claude Sonnet) on every search call.

## Quick Start: Search the Pre-Built Index

Three videos are already ingested and indexed (`video_1`, `video_12`, `video_36` — see [Video Corpus](#video-corpus)). No ingestion needed.

```bash
# 1. Build the search index from pre-built metadata
videosearch index

# 2. Search
videosearch search "Find all interactions where an officer reads Miranda rights"
videosearch search "Find every moment where someone raises their voice"

# 3. Run all 6 example queries at once
videosearch batch-eval

# 4. Optional: start the API server
videosearch serve
# then open http://127.0.0.1:8000/docs
```

## Full Pipeline: Ingest Your Own Videos

```bash
# Estimate Gemini captioning cost before committing
videosearch estimate data/videos/your_video.mp4

# Ingest without visual captioning (transcript + audio + OCR only, no API cost)
videosearch ingest data/videos/your_video.mp4

# Ingest with visual captioning (uses Gemini Flash via OpenRouter)
videosearch ingest data/videos/your_video.mp4 --caption

# Rebuild the index after ingestion
videosearch index

# Then search as normal
videosearch search "Find all license plates visible in the footage"
```

**First-run notes:**
- `faster-whisper` downloads the `large-v3` model (~3 GB) on first transcription call. Subsequent runs use the cached model.
- Ingestion runs transcription, audio analysis, OCR, and optional captioning in parallel per chunk. A 30-minute video takes roughly 10–20 minutes on CPU.
- Set `WHISPER_COMPUTE_TYPE=int8` in `.env` to force CPU-optimized inference if you don't have CUDA.

## Video Corpus

| Video ID  | Duration | Description |
|-----------|----------|-------------|
| video_1   | ~33 min  | Traffic stop / DUI arrest — Miranda rights, implied consent, breathalyzer |
| video_12  | ~14 min  | Emergency response — fire/EMS scene, hospital transport |
| video_36  | ~24 min  | Traffic stop / drug investigation — pursuit, field sobriety, breathalyzer |

## Commands

| Command | Description |
|---------|-------------|
| `videosearch ingest <video>` | Process video through extraction pipeline |
| `videosearch ingest <video> --caption` | Ingest with Gemini visual captioning |
| `videosearch index` | Build search indices from metadata |
| `videosearch search <query>` | Search with natural language query |
| `videosearch batch-eval` | Run all 6 example queries |
| `videosearch estimate <video>` | Show captioning cost estimate |
| `videosearch caption <video>` | Generate visual captions for an ingested video |
| `videosearch serve` | Start FastAPI server |

## Docker

```bash
docker build -t videosearch .
docker run --env-file .env videosearch videosearch batch-eval
```

## Architecture

Three-layer pipeline:

**Extraction** (run once per video)
- Compression: FFmpeg → 720p H.264
- Chunking: PySceneDetect scene boundaries with sliding-window fallback (~30s chunks)
- Transcription: faster-whisper `large-v3` — word-level timestamps, VAD + hallucination filtering
- Audio analysis: librosa — RMS energy, pitch, zero-crossing rate, raised-voice detection
- OCR: PaddleOCR on sampled frames (1 per 2s), EasyOCR fallback
- Captioning: Gemini 2.5 Flash — dense visual description per chunk (optional, uses API credits)

**Index**
- Embedder: Gemini `gemini-embedding-001` (768-dim); falls back to `all-MiniLM-L6-v2` (384-dim) on quota exhaustion
- Vector store: LanceDB (embedded, file-based)
- Keyword store: BM25 over raw transcripts (rank-bm25)
- Metadata filters: structured fields (time-of-day, volume level, has_ocr)

**Query**
- Classifier: Gemini Flash via OpenRouter — routes query to one of five types (visual, audio, transcript, temporal, mixed) and sets RRF fusion weights
- Retriever: hybrid search — vector + BM25 + filter results fused via Reciprocal Rank Fusion
- Reranker: Claude Sonnet via OpenRouter — scores top-K candidates and returns per-result reasoning
