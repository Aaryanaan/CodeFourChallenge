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
    """ingest /nonexistent.mp4 prints Error and exits with code 1."""
    result = runner.invoke(app, ["ingest", "/nonexistent_video_file_abc123.mp4"])

    assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}: {result.output}"
    assert "Error" in result.output or "not found" in result.output


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
        with patch("videosearch.cli.Settings"):
            result = runner.invoke(app, ["search", "Miranda rights"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "Results for" in result.output
    assert "bodycam_001" in result.output


def test_search_command_no_results():
    """search with no results prints 'No results found' and exits 0."""
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.retrieve.return_value = []

    with patch("videosearch.cli.HybridRetriever", return_value=mock_retriever_instance):
        with patch("videosearch.cli.Settings"):
            result = runner.invoke(app, ["search", "nonexistent_query_xyz"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "No results found" in result.output


def test_search_command_top_k():
    """search --top-k 5 passes top_k=5 to HybridRetriever.retrieve()."""
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.retrieve.return_value = MOCK_RESULTS

    with patch("videosearch.cli.HybridRetriever", return_value=mock_retriever_instance):
        with patch("videosearch.cli.Settings"):
            result = runner.invoke(app, ["search", "some query", "--top-k", "5"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    mock_retriever_instance.retrieve.assert_called_once_with("some query", top_k=5)


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

    with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
        with patch("videosearch.cli.GeminiCaptioner", return_value=mock_captioner_instance):
            with patch("videosearch.cli.Settings") as mock_settings_cls:
                mock_settings = MagicMock()
                mock_settings.video_dir = Path("data/videos")
                mock_settings_cls.return_value = mock_settings

                result = runner.invoke(app, ["caption", "test_video.mp4"])

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
    assert "Captioned" in result.output
    assert "3 new" in result.output
    mock_writer_instance.write.assert_called_once()
    # Verify each chunk got visual_caption set
    for chunk in chunks:
        assert chunk.visual_caption == "Clothing: uniform"


def test_caption_command_no_metadata():
    """caption <video> exits 1 when metadata not found."""
    mock_writer_instance = MagicMock()
    mock_writer_instance.load.side_effect = FileNotFoundError("no metadata")

    with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
        with patch("videosearch.cli.GeminiCaptioner"):
            with patch("videosearch.cli.Settings"):
                result = runner.invoke(app, ["caption", "test_video.mp4"])

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

    with patch("videosearch.cli.MetadataWriter", return_value=mock_writer_instance):
        with patch("videosearch.cli.GeminiCaptioner", return_value=mock_captioner_instance):
            with patch("videosearch.cli.Settings") as mock_settings_cls:
                mock_settings = MagicMock()
                mock_settings.video_dir = Path("data/videos")
                mock_settings_cls.return_value = mock_settings

                result = runner.invoke(app, ["caption", "test_video.mp4"])

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
