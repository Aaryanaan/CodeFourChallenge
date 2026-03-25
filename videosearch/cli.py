"""Typer CLI for video search pipeline.

Commands:
  ingest <video>   -- Process a video through IngestionPipeline
  index [ids...]   -- Build search indices via IndexBuilder
  search <query>   -- Retrieve results via HybridRetriever
  estimate <video> -- Show estimated captioning cost (ING-09)
  caption <video>  -- Generate visual captions for chunk metadata

Heavy pipeline classes are imported at module level so they can be patched
in tests via `patch("videosearch.cli.IngestionPipeline", ...)`.
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

from videosearch.captioner import GeminiCaptioner
from videosearch.config import Settings
from videosearch.hybrid_retriever import HybridRetriever
from videosearch.index_builder import IndexBuilder
from videosearch.ingestion import IngestionPipeline
from videosearch.metadata_writer import MetadataWriter

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
    try:
        results = retriever.retrieve(query, top_k=top_k)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit()

    _print_results(results, query)


@app.command()
def estimate(
    video: str = typer.Argument(..., help="Path to video file"),
) -> None:
    """Show estimated captioning cost before any API call (ING-09)."""
    settings = Settings()
    video_id = Path(video).stem
    writer = MetadataWriter(metadata_dir=settings.metadata_dir)

    try:
        chunks = writer.load(video_id)
        chunk_count = len(chunks)
        # Count already-cached chunks (per D-11)
        cached_count = sum(
            1 for c in chunks
            if (settings.caption_cache_dir / video_id / f"{c.chunk_index}.json").exists()
        )
    except FileNotFoundError:
        # Fallback: estimate from video duration (D-16)
        import subprocess as _sp
        result = _sp.run(
            [settings.ffmpeg_path.replace("ffmpeg", "ffprobe"),
             "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video],
            capture_output=True, text=True,
        )
        duration = float(result.stdout.strip()) if result.stdout.strip() else 0.0
        chunk_count = max(1, int(duration / 30))
        cached_count = 0

    new_chunks = chunk_count - cached_count
    cost_per_chunk = settings.caption_cost_per_chunk
    total_cost = new_chunks * cost_per_chunk
    ceiling = settings.caption_cost_ceiling

    # Rich table output (D-17)
    table = Table(title=f"Cost Estimate: {video_id}")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total chunks", str(chunk_count))
    table.add_row("Already cached", str(cached_count))
    table.add_row("New (need API call)", str(new_chunks))
    table.add_row("Cost per chunk", f"${cost_per_chunk:.4f}")
    table.add_row("Estimated cost", f"${total_cost:.4f}")
    table.add_row("Budget ceiling", f"${ceiling:.2f}")
    table.add_row("Ceiling usage", f"{(total_cost / ceiling * 100):.1f}%")
    console.print(table)

    # D-19: warn if exceeds ceiling
    if total_cost > ceiling:
        console.print(
            f"[red]Warning:[/red] Estimated cost ${total_cost:.4f} exceeds "
            f"ceiling ${ceiling:.2f}. Aborting."
        )
        raise typer.Exit(code=1)


@app.command()
def caption(
    video: str = typer.Argument(..., help="Path to video file"),
) -> None:
    """Generate visual captions for an ingested video's chunks."""
    settings = Settings()
    video_id = Path(video).stem
    writer = MetadataWriter(metadata_dir=settings.metadata_dir)

    try:
        chunks = writer.load(video_id)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] No metadata for {video_id}. Run `ingest` first.")
        raise typer.Exit(code=1)

    compressed_path = str(settings.video_dir / "compressed" / f"{video_id}_720p.mp4")
    if not Path(compressed_path).exists():
        console.print(
            f"[red]Error:[/red] Compressed video not found: {compressed_path}\n"
            f"Run `ingest {video}` to generate it."
        )
        raise typer.Exit(code=1)

    # Cost preflight: enforce ceiling before any API spend
    _new_count = sum(
        1 for c in chunks
        if not (settings.caption_cache_dir / video_id / f"{c.chunk_index}.json").exists()
    )
    _cost = _new_count * settings.caption_cost_per_chunk
    if _cost > settings.caption_cost_ceiling:
        console.print(
            f"[red]Error:[/red] Estimated cost ${_cost:.4f} exceeds ceiling "
            f"${settings.caption_cost_ceiling:.2f}. Run `estimate {video}` for details."
        )
        raise typer.Exit(code=1)

    captioner = GeminiCaptioner(settings)
    cached_count = 0
    fresh_count = 0
    failed_count = 0

    for chunk in chunks:
        try:
            result = captioner.caption(compressed_path, chunk.start_time, chunk.end_time, chunk.chunk_index)
            chunk.visual_caption = result["caption"]
            if result.get("cached", False):
                cached_count += 1
            else:
                fresh_count += 1
        except Exception as exc:
            # D-24: same pattern as all other extractors — log warning, continue.
            # visual_caption stays None for this chunk. No pipeline abort.
            failed_count += 1
            console.print(
                f"[yellow]Warning:[/yellow] Caption failed for chunk "
                f"{chunk.chunk_index}: {exc}"
            )

    # Always write, even if some chunks failed — preserves partial progress
    writer.write(video_id, chunks)
    summary = f"({fresh_count} new, {cached_count} cached)"
    if failed_count:
        summary = f"({fresh_count} new, {cached_count} cached, {failed_count} failed)"
    console.print(f"[green]Captioned:[/green] {video_id} {summary}")


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
