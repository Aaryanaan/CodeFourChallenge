"""Tests for the Typer CLI app (videosearch/cli.py).

Tests use CliRunner to invoke CLI commands with mocked backends.
All heavy dependencies (IngestionPipeline, IndexBuilder, HybridRetriever)
are patched so tests run without real video files or API keys.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from videosearch.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# ingest command
# ---------------------------------------------------------------------------


def test_ingest_command_success():
    """ingest <video> calls IngestionPipeline.ingest() and prints video_id."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.ingest.return_value = "test_video"

    with patch("videosearch.cli.IngestionPipeline", return_value=mock_pipeline_instance) as mock_pipeline_cls:
        with patch("videosearch.cli.Settings") as mock_settings_cls:
            result = runner.invoke(app, ["ingest", tmp_path])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "Ingested" in result.output
    assert "test_video" in result.output

    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)


def test_ingest_command_missing_file():
    """ingest /nonexistent.mp4 prints FAILED and exits with code 1."""
    result = runner.invoke(app, ["ingest", "/nonexistent_video_file_abc123.mp4"])

    assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}: {result.output}"
    assert "FAILED" in result.output or "not found" in result.output


def test_ingest_multi_video_success():
    """ingest with 2 videos succeeds and reports '2 succeeded'."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f2:
        tmp1, tmp2 = f1.name, f2.name

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.ingest.side_effect = ["video_001", "video_002"]

    with patch("videosearch.cli.IngestionPipeline", return_value=mock_pipeline_instance):
        with patch("videosearch.cli.Settings"):
            result = runner.invoke(app, ["ingest", tmp1, tmp2])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "2 succeeded" in result.output
    assert mock_pipeline_instance.ingest.call_count == 2

    Path(tmp1).unlink(missing_ok=True)
    Path(tmp2).unlink(missing_ok=True)


def test_ingest_multi_video_partial_failure():
    """ingest with 1 existing + 1 nonexistent video reports partial failure."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_exists = f.name

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.ingest.return_value = "video_001"

    with patch("videosearch.cli.IngestionPipeline", return_value=mock_pipeline_instance):
        with patch("videosearch.cli.Settings"):
            result = runner.invoke(app, ["ingest", tmp_exists, "/nonexistent.mp4"])

    assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}: {result.output}"
    assert "1 succeeded" in result.output
    assert "1 failed" in result.output

    Path(tmp_exists).unlink(missing_ok=True)


def test_ingest_caption_flag():
    """ingest --caption passes include_caption=True to pipeline."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.ingest.return_value = "test_video"

    with patch("videosearch.cli.IngestionPipeline", return_value=mock_pipeline_instance):
        with patch("videosearch.cli.Settings"):
            result = runner.invoke(app, ["ingest", tmp_path, "--caption"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    mock_pipeline_instance.ingest.assert_called_once_with(str(Path(tmp_path)), include_caption=True)

    Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------


def test_index_command_success():
    """index command with mocked IndexBuilder prints 'Indexed'."""
    mock_builder_instance = MagicMock()
    mock_builder_instance.build_index.return_value = {
        "total_chunks": 10,
        "embedded_chunks": 8,
        "skipped_chunks": 2,
        "skipped_videos": 0,
        "bm25_indexed": 10,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake JSON metadata file so index discovers something
        (Path(tmpdir) / "video_001.json").write_text("{}")

        with patch("videosearch.cli.IndexBuilder", return_value=mock_builder_instance):
            with patch("videosearch.cli.Settings") as mock_settings_cls:
                mock_settings_instance = MagicMock()
                mock_settings_instance.metadata_dir = Path(tmpdir)
                mock_settings_cls.return_value = mock_settings_instance

                result = runner.invoke(app, ["index"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "Indexed" in result.output


def test_index_force_flag():
    """index --force passes force=True to builder.build_index."""
    mock_builder_instance = MagicMock()
    mock_builder_instance.build_index.return_value = {
        "total_chunks": 10,
        "embedded_chunks": 10,
        "skipped_chunks": 0,
        "skipped_videos": 0,
        "bm25_indexed": 10,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "video_001.json").write_text("{}")

        with patch("videosearch.cli.IndexBuilder", return_value=mock_builder_instance):
            with patch("videosearch.cli.Settings") as mock_settings_cls:
                mock_settings_instance = MagicMock()
                mock_settings_instance.metadata_dir = Path(tmpdir)
                mock_settings_cls.return_value = mock_settings_instance

                result = runner.invoke(app, ["index", "--force"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    mock_builder_instance.build_index.assert_called_once()
    call_kwargs = mock_builder_instance.build_index.call_args
    assert call_kwargs[1].get("force") is True or (len(call_kwargs[0]) > 1 and call_kwargs[0][1] is True)


def test_index_command_no_metadata():
    """index command with empty metadata dir prints 'No metadata found' and exits 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # tmpdir has no .json files
        with patch("videosearch.cli.Settings") as mock_settings_cls:
            mock_settings_instance = MagicMock()
            mock_settings_instance.metadata_dir = Path(tmpdir)
            mock_settings_cls.return_value = mock_settings_instance

            result = runner.invoke(app, ["index"])

    assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}: {result.output}"
    assert "No metadata found" in result.output


# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------


MOCK_RESULTS = [
    {
        "video_id": "bodycam_001",
        "chunk_index": 3,
        "start_time": 30.0,
        "end_time": 60.0,
        "combined_text": "Miranda rights read aloud to suspect",
        "rrf_score": 0.016,
    },
    {
        "video_id": "bodycam_002",
        "chunk_index": 1,
        "start_time": 10.0,
        "end_time": 40.0,
        "combined_text": "Officer reading rights at scene",
        "rrf_score": 0.012,
    },
]


def test_search_command_success():
    """search 'Miranda rights' with mock retriever returns results table."""
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.retrieve.return_value = MOCK_RESULTS

    with patch("videosearch.cli.HybridRetriever", return_value=mock_retriever_instance):
        with patch("videosearch.cli.GeminiQueryClassifier"):
            with patch("videosearch.cli.ClaudeReranker"):
                with patch("videosearch.cli.Settings"):
                    result = runner.invoke(app, ["search", "Miranda rights"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "Results for" in result.output
    # Video ID may be truncated by Rich table in narrow terminal
    assert "bodycam" in result.output or "bodyc" in result.output


def test_search_command_no_results():
    """search with no results prints 'No results found' and exits 0."""
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.retrieve.return_value = []

    with patch("videosearch.cli.HybridRetriever", return_value=mock_retriever_instance):
        with patch("videosearch.cli.GeminiQueryClassifier"):
            with patch("videosearch.cli.ClaudeReranker"):
                with patch("videosearch.cli.Settings"):
                    result = runner.invoke(app, ["search", "nonexistent_query_xyz"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "No results found" in result.output


def test_search_command_top_k():
    """search --top-k 5 passes top_k=5 to HybridRetriever.retrieve()."""
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.retrieve.return_value = MOCK_RESULTS

    with patch("videosearch.cli.HybridRetriever", return_value=mock_retriever_instance):
        with patch("videosearch.cli.GeminiQueryClassifier"):
            with patch("videosearch.cli.ClaudeReranker"):
                with patch("videosearch.cli.Settings"):
                    result = runner.invoke(app, ["search", "some query", "--top-k", "5"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    mock_retriever_instance.retrieve.assert_called_once_with("some query", top_k=5)


@patch("videosearch.cli.HybridRetriever")
@patch("videosearch.cli.ClaudeReranker")
@patch("videosearch.cli.GeminiQueryClassifier")
def test_search_wires_classifier_and_reranker(mock_classifier_cls, mock_reranker_cls, mock_retriever_cls):
    """search command injects classifier and reranker into HybridRetriever."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"video_id": "v1", "chunk_index": 0, "start_time": 0.0, "end_time": 10.0,
         "combined_text": "test", "rrf_score": 0.5, "reasoning": "relevant"}
    ]
    mock_retriever_cls.return_value = mock_retriever

    with patch("videosearch.cli.Settings"):
        result = runner.invoke(app, ["search", "test query"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    # Verify HybridRetriever was constructed with classifier and reranker kwargs
    mock_retriever_cls.assert_called_once()
    call_kwargs = mock_retriever_cls.call_args
    assert "classifier" in str(call_kwargs), f"classifier not passed to HybridRetriever: {call_kwargs}"
    assert "reranker" in str(call_kwargs), f"reranker not passed to HybridRetriever: {call_kwargs}"
    # Verify GeminiQueryClassifier and ClaudeReranker were instantiated
    mock_classifier_cls.assert_called_once()
    mock_reranker_cls.assert_called_once()


# ---------------------------------------------------------------------------
# estimate command
# ---------------------------------------------------------------------------


def _make_mock_chunks(count: int, video_id: str = "test_video") -> list:
    """Create minimal MagicMock chunks for testing."""
    from videosearch.models import ChunkMetadata

    chunks = []
    for i in range(count):
        chunk = MagicMock(spec=ChunkMetadata)
        chunk.chunk_index = i
        chunk.visual_caption = None
        chunks.append(chunk)
    return chunks


def test_estimate_command():
    """estimate <video> with 10 chunks (3 cached) shows cost table."""
    chunks = _make_mock_chunks(10)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cache_dir = tmp_path / "captions"
        video_id = "test_video"

        # Create 3 cache files to simulate already-cached chunks
        (cache_dir / video_id).mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (cache_dir / video_id / f"{i}.json").write_text('{"caption":"x"}')

        mock_writer_instance = MagicMock()
        mock_writer_instance.load.return_value = chunks

        with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
            with patch("videosearch.cli.Settings") as mock_settings_cls:
                mock_settings = MagicMock()
                mock_settings.caption_cost_per_chunk = 0.003
                mock_settings.caption_cost_ceiling = 5.0
                mock_settings.caption_cache_dir = cache_dir
                mock_settings.metadata_dir = tmp_path / "metadata"
                mock_settings_cls.return_value = mock_settings

                result = runner.invoke(app, ["estimate", "test_video.mp4"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "10" in result.output  # total chunks
    assert "3" in result.output   # cached count
    assert "7" in result.output   # new chunks


def test_estimate_fallback_no_metadata():
    """estimate <video> falls back to ffprobe when metadata missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mock_writer_instance = MagicMock()
        mock_writer_instance.load.side_effect = FileNotFoundError("no metadata")

        with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
            with patch("videosearch.cli.Settings") as mock_settings_cls:
                mock_settings = MagicMock()
                mock_settings.caption_cost_per_chunk = 0.003
                mock_settings.caption_cost_ceiling = 5.0
                mock_settings.caption_cache_dir = tmp_path / "captions"
                mock_settings.metadata_dir = tmp_path / "metadata"
                mock_settings.ffmpeg_path = "ffmpeg"
                mock_settings_cls.return_value = mock_settings

                mock_proc = MagicMock()
                mock_proc.stdout = "300.0\n"

                with patch("subprocess.run", return_value=mock_proc):
                    result = runner.invoke(app, ["estimate", "test_video.mp4"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    # 300s / 30s per chunk = 10 chunks
    assert "10" in result.output


def test_estimate_exceeds_ceiling():
    """estimate warns and exits 1 when cost exceeds ceiling."""
    chunks = _make_mock_chunks(2000)  # 2000 * 0.003 = $6.00 > $5.00

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mock_writer_instance = MagicMock()
        mock_writer_instance.load.return_value = chunks

        with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
            with patch("videosearch.cli.Settings") as mock_settings_cls:
                mock_settings = MagicMock()
                mock_settings.caption_cost_per_chunk = 0.003
                mock_settings.caption_cost_ceiling = 5.0
                mock_settings.caption_cache_dir = tmp_path / "captions"
                mock_settings.metadata_dir = tmp_path / "metadata"
                mock_settings_cls.return_value = mock_settings

                result = runner.invoke(app, ["estimate", "test_video.mp4"])

    assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}: {result.output}"
    assert "exceeds" in result.output


# ---------------------------------------------------------------------------
# caption command
# ---------------------------------------------------------------------------


def test_caption_command_success():
    """caption <video> captions all chunks and writes metadata."""
    from videosearch.models import ChunkMetadata

    chunks = [
        MagicMock(spec=ChunkMetadata, chunk_index=i, visual_caption=None,
                  start_time=float(i * 30), end_time=float((i + 1) * 30))
        for i in range(3)
    ]

    mock_writer_instance = MagicMock()
    mock_writer_instance.load.return_value = chunks

    mock_captioner_instance = MagicMock()
    mock_captioner_instance.caption.return_value = {"caption": "Clothing: uniform", "cached": False}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        compressed_dir = tmp_path / "compressed"
        compressed_dir.mkdir()
        compressed_file = compressed_dir / "test_video_720p.mp4"
        compressed_file.write_bytes(b"")  # exists check only

        with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
            with patch("videosearch.cli.GeminiCaptioner", return_value=mock_captioner_instance):
                with patch("videosearch.cli.Settings") as mock_settings_cls:
                    mock_settings = MagicMock()
                    mock_settings.video_dir = tmp_path
                    mock_settings.caption_cache_dir = tmp_path / "captions"
                    mock_settings.caption_cost_per_chunk = 0.003
                    mock_settings.caption_cost_ceiling = 5.0
                    mock_settings_cls.return_value = mock_settings

                    result = runner.invoke(app, ["caption", "test_video.mp4", "--yes"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "Captioned" in result.output
    assert "3 new" in result.output
    mock_writer_instance.write.assert_called_once()
    for chunk in chunks:
        assert chunk.visual_caption == "Clothing: uniform"


def test_caption_command_compressed_missing():
    """caption exits 1 with clear error when compressed file does not exist."""
    from videosearch.models import ChunkMetadata

    chunks = [MagicMock(spec=ChunkMetadata, chunk_index=0, visual_caption=None,
                        start_time=0.0, end_time=30.0)]
    mock_writer_instance = MagicMock()
    mock_writer_instance.load.return_value = chunks

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # compressed/ dir exists but the file does NOT
        (tmp_path / "compressed").mkdir()

        with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
            with patch("videosearch.cli.GeminiCaptioner"):
                with patch("videosearch.cli.Settings") as mock_settings_cls:
                    mock_settings = MagicMock()
                    mock_settings.video_dir = tmp_path
                    mock_settings.caption_cache_dir = tmp_path / "captions"
                    mock_settings.caption_cost_per_chunk = 0.003
                    mock_settings.caption_cost_ceiling = 5.0
                    mock_settings_cls.return_value = mock_settings

                    result = runner.invoke(app, ["caption", "test_video.mp4", "--yes"])

    assert result.exit_code == 1
    assert "Compressed video not found" in result.output


def test_caption_command_exceeds_ceiling():
    """caption exits 1 when estimated cost exceeds ceiling."""
    from videosearch.models import ChunkMetadata

    # 2000 uncached chunks * $0.003 = $6.00 > $5.00 ceiling
    chunks = [MagicMock(spec=ChunkMetadata, chunk_index=i, visual_caption=None,
                        start_time=float(i * 30), end_time=float((i + 1) * 30))
              for i in range(2000)]
    mock_writer_instance = MagicMock()
    mock_writer_instance.load.return_value = chunks

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        compressed_dir = tmp_path / "compressed"
        compressed_dir.mkdir()
        (compressed_dir / "test_video_720p.mp4").write_bytes(b"")

        with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
            with patch("videosearch.cli.GeminiCaptioner"):
                with patch("videosearch.cli.Settings") as mock_settings_cls:
                    mock_settings = MagicMock()
                    mock_settings.video_dir = tmp_path
                    mock_settings.caption_cache_dir = tmp_path / "captions"
                    mock_settings.caption_cost_per_chunk = 0.003
                    mock_settings.caption_cost_ceiling = 5.0
                    mock_settings_cls.return_value = mock_settings

                    result = runner.invoke(app, ["caption", "test_video.mp4", "--yes"])

    assert result.exit_code == 1
    assert "exceeds ceiling" in result.output


def test_caption_command_no_metadata():
    """caption <video> exits 1 when metadata not found."""
    mock_writer_instance = MagicMock()
    mock_writer_instance.load.side_effect = FileNotFoundError("no metadata")

    with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
        with patch("videosearch.cli.GeminiCaptioner"):
            with patch("videosearch.cli.Settings"):
                result = runner.invoke(app, ["caption", "test_video.mp4", "--yes"])

    assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}: {result.output}"
    assert "No metadata" in result.output


def test_caption_command_partial_failure():
    """caption continues on chunk failure (D-24), saves partial progress."""
    from videosearch.models import ChunkMetadata

    chunks = [
        MagicMock(spec=ChunkMetadata, chunk_index=i, visual_caption=None,
                  start_time=float(i * 30), end_time=float((i + 1) * 30))
        for i in range(3)
    ]

    mock_writer_instance = MagicMock()
    mock_writer_instance.load.return_value = chunks

    mock_captioner_instance = MagicMock()
    mock_captioner_instance.caption.side_effect = [
        {"caption": "Clothing: uniform", "cached": False},  # chunk 0: success
        Exception("API timeout"),                             # chunk 1: fail
        {"caption": "Actions: walking", "cached": False},    # chunk 2: success
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        compressed_dir = tmp_path / "compressed"
        compressed_dir.mkdir()
        (compressed_dir / "test_video_720p.mp4").write_bytes(b"")

        with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
            with patch("videosearch.cli.GeminiCaptioner", return_value=mock_captioner_instance):
                with patch("videosearch.cli.Settings") as mock_settings_cls:
                    mock_settings = MagicMock()
                    mock_settings.video_dir = tmp_path
                    mock_settings.caption_cache_dir = tmp_path / "captions"
                    mock_settings.caption_cost_per_chunk = 0.003
                    mock_settings.caption_cost_ceiling = 5.0
                    mock_settings_cls.return_value = mock_settings

                    result = runner.invoke(app, ["caption", "test_video.mp4", "--yes"])

    # No pipeline abort -- exit 0
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    # Yellow warning for failed chunk
    assert "Warning" in result.output
    assert "API timeout" in result.output
    # Summary shows 2 new, 1 failed
    assert "2 new" in result.output
    assert "1 failed" in result.output
    # writer.write always called (partial progress saved)
    mock_writer_instance.write.assert_called_once()
    # First and third chunks got captions; second did not
    assert chunks[0].visual_caption == "Clothing: uniform"
    assert chunks[1].visual_caption is None
    assert chunks[2].visual_caption == "Actions: walking"


def test_caption_multi_video_cumulative_budget():
    """caption with 2 videos shows cumulative spend after each."""
    from videosearch.models import ChunkMetadata

    chunks1 = [MagicMock(spec=ChunkMetadata, chunk_index=i, visual_caption=None,
                         start_time=float(i * 30), end_time=float((i + 1) * 30))
               for i in range(2)]
    chunks2 = [MagicMock(spec=ChunkMetadata, chunk_index=i, visual_caption=None,
                         start_time=float(i * 30), end_time=float((i + 1) * 30))
               for i in range(3)]

    mock_writer_instance = MagicMock()
    mock_writer_instance.load.side_effect = [chunks1, chunks2]

    mock_captioner_instance = MagicMock()
    mock_captioner_instance.caption.return_value = {"caption": "test caption", "cached": False}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        compressed_dir = tmp_path / "compressed"
        compressed_dir.mkdir()
        (compressed_dir / "vid1_720p.mp4").write_bytes(b"")
        (compressed_dir / "vid2_720p.mp4").write_bytes(b"")

        with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
            with patch("videosearch.cli.GeminiCaptioner", return_value=mock_captioner_instance):
                with patch("videosearch.cli.Settings") as mock_settings_cls:
                    mock_settings = MagicMock()
                    mock_settings.video_dir = tmp_path
                    mock_settings.caption_cache_dir = tmp_path / "captions"
                    mock_settings.caption_cost_per_chunk = 0.003
                    mock_settings.caption_cost_ceiling = 5.0
                    mock_settings_cls.return_value = mock_settings

                    result = runner.invoke(app, ["caption", "vid1.mp4", "vid2.mp4", "--yes"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "Spent:" in result.output
    assert "2 succeeded" in result.output
