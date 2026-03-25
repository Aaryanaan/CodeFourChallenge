"""Typer CLI for video search pipeline.

Commands:
  ingest <video>   -- Process a video through IngestionPipeline
  index [ids...]   -- Build search indices via IndexBuilder
  search <query>   -- Retrieve results via HybridRetriever

Heavy pipeline classes are imported at module level so they can be patched
in tests via `patch("videosearch.cli.IngestionPipeline", ...)`.
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

from videosearch.config import Settings
from videosearch.hybrid_retriever import HybridRetriever
from videosearch.index_builder import IndexBuilder
from videosearch.ingestion import IngestionPipeline

app = typer.Typer(name="videosearch", help="Search body-worn camera footage")
console = Console()


@app.command()
def ingest(
    video: str = typer.Argument(..., help="Path to video file"),
) -> None:
    """Process a video file through the full ingestion pipeline."""
    if not Path(video).exists():
        console.print(f"[red]Error:[/red] Video not found: {video}")
        raise typer.Exit(code=1)

    settings = Settings()
    pipeline = IngestionPipeline(settings)
    video_id = pipeline.ingest(str(Path(video)))
    console.print(f"[green]Ingested:[/green] {video_id}")


@app.command()
def index(
    video_ids: list[str] = typer.Argument(None, help="Video IDs to index (default: all)"),
) -> None:
    """Build search indices from ingested metadata."""
    settings = Settings()

    if not video_ids:
        video_ids = [p.stem for p in settings.metadata_dir.glob("*.json")]

    if not video_ids:
        console.print("[red]Error:[/red] No metadata found. Run `ingest` first.")
        raise typer.Exit(code=1)

    builder = IndexBuilder(settings)
    stats = builder.build_index(video_ids)
    console.print(
        f"[green]Indexed:[/green] {stats['total_chunks']} chunks "
        f"({stats['embedded_chunks']} embedded, {stats['bm25_indexed']} BM25)"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
) -> None:
    """Search ingested video footage with a natural language query."""
    settings = Settings()
    retriever = HybridRetriever(settings)
    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit()

    _print_results(results, query)


def _print_results(results: list[dict], query: str) -> None:
    """Render search results as a Rich table."""
    table = Table(title=f"Results for: {query}")
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Video", width=20, no_wrap=True)
    table.add_column("Time", width=15)
    table.add_column("Score", width=8)
    table.add_column("Snippet", width=60)

    for rank, r in enumerate(results, start=1):
        start = f"{r['start_time']:.1f}s"
        end = f"{r['end_time']:.1f}s"
        time_range = f"{start} – {end}"
        snippet = r.get("combined_text", "")[:80]
        score = f"{r.get('rrf_score', 0):.4f}"
        table.add_row(str(rank), r["video_id"], time_range, score, snippet)

    console.print(table)


if __name__ == "__main__":
    app()
