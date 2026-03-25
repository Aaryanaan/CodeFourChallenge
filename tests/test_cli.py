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
