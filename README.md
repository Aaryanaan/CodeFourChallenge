# VideoSearch

Natural language search for body-worn camera footage.

## Installation

```bash
pip install -e ".[dev]"
```

## Demo

### 1. Ingest video(s)

```bash
videosearch ingest data/videos/your_video.mp4
```

With visual captioning (uses Gemini API):
```bash
videosearch ingest data/videos/your_video.mp4 --caption
```

### 2. Build search index

```bash
videosearch index
```

### 3. Search

```bash
videosearch search "Find all interactions where an officer reads Miranda rights"
videosearch search "Find every moment where someone raises their voice"
videosearch search "Locate all footage containing a person in a red shirt"
```

### 4. Batch evaluation (all 6 example queries)

```bash
videosearch batch-eval
```

### 5. API server (optional)

```bash
videosearch serve
```

Then open http://127.0.0.1:8000/docs for the Swagger UI.

## Environment Variables

Create a `.env` file:

```
GOOGLE_API_KEY=your-google-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

## Commands

| Command | Description |
|---------|-------------|
| `videosearch ingest <video>` | Process video through extraction pipeline |
| `videosearch index` | Build search indices from metadata |
| `videosearch search <query>` | Search with natural language query |
| `videosearch estimate <video>` | Show captioning cost estimate |
| `videosearch caption <video>` | Generate visual captions |
| `videosearch batch-eval` | Run all 6 example queries |
| `videosearch serve` | Start FastAPI server |
