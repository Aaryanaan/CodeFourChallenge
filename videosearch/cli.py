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
from videosearch.classifier import GeminiQueryClassifier
from videosearch.config import Settings
from videosearch.hybrid_retriever import HybridRetriever
from videosearch.index_builder import IndexBuilder
from videosearch.ingestion import IngestionPipeline
from videosearch.metadata_writer import MetadataWriter
from videosearch.reranker import ClaudeReranker

app = typer.Typer(name="videosearch", help="Search body-worn camera footage")
console = Console()


@app.command()
def ingest(
    videos: list[str] = typer.Argument(..., help="Path(s) to video file(s)"),
    caption: bool = typer.Option(False, "--caption", help="Include visual captioning (per D-02)"),
) -> None:
    """Process video file(s) through the full ingestion pipeline."""
    settings = Settings()
    pipeline = IngestionPipeline(settings)
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for video in videos:
        if not Path(video).exists():
            failed.append((video, "File not found"))
            console.print(f"  [red]FAILED {Path(video).name}: File not found[/red]")
            continue
        try:
            video_id = pipeline.ingest(str(Path(video)), include_caption=caption)
            succeeded.append(video_id)
            console.print(f"  [{len(succeeded)}/{len(videos)}] [green]Ingested: {video_id}[/green]")
        except Exception as e:
            failed.append((video, str(e)))
            console.print(f"  [red]FAILED {Path(video).name}: {e}[/red]")

    # Summary (per D-10)
    console.print(f"\n[green]Done:[/green] {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        for path, err in failed:
            console.print(f"  [red]-[/red] {path}: {err}")
        raise typer.Exit(code=1)


@app.command()
def index(
    video_ids: list[str] = typer.Argument(None, help="Video IDs to index (default: all)"),
    force: bool = typer.Option(False, "--force", help="Re-embed all videos (ignore cache)"),
) -> None:
    """Build search indices from ingested metadata."""
    settings = Settings()

    if not video_ids:
        video_ids = [p.stem for p in settings.metadata_dir.glob("*.json")]

    if not video_ids:
        console.print("[red]Error:[/red] No metadata found. Run `ingest` first.")
        raise typer.Exit(code=1)

    builder = IndexBuilder(settings)
    stats = builder.build_index(video_ids, force=force)
    console.print(
        f"[green]Indexed:[/green] {stats['total_chunks']} chunks "
        f"({stats['embedded_chunks']} embedded, {stats.get('skipped_videos', 0)} videos skipped, "
        f"{stats['bm25_indexed']} BM25)"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
) -> None:
    """Search ingested video footage with a natural language query."""
    settings = Settings()
    classifier = GeminiQueryClassifier(settings)
    reranker = ClaudeReranker(settings)
    retriever = HybridRetriever(settings, classifier=classifier, reranker=reranker)
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
    videos: list[str] = typer.Argument(..., help="Path(s) to video file(s)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Generate visual captions for ingested video(s) chunks."""
    settings = Settings()
    writer = MetadataWriter(metadata_dir=settings.metadata_dir)
    captioner = GeminiCaptioner(settings)

    succeeded: list[str] = []
    failed_videos: list[tuple[str, str]] = []
    cumulative_spend = 0.0

    for idx, video in enumerate(videos, 1):
        video_id = Path(video).stem

        try:
            chunks = writer.load(video_id)
        except FileNotFoundError:
            failed_videos.append((video, f"No metadata for {video_id}"))
            console.print(f"  [red]FAILED {video_id}: No metadata. Run `ingest` first.[/red]")
            continue

        compressed_path = str(settings.video_dir / "compressed" / f"{video_id}_720p.mp4")
        if not Path(compressed_path).exists():
            failed_videos.append((video, "Compressed video not found"))
            console.print(f"  [red]FAILED {video_id}: Compressed video not found[/red]")
            continue

        # Cost preflight: enforce ceiling before any API spend
        _new_count = sum(
            1 for c in chunks
            if not (settings.caption_cache_dir / video_id / f"{c.chunk_index}.json").exists()
        )
        _cost = _new_count * settings.caption_cost_per_chunk
        if _cost > settings.caption_cost_ceiling:
            failed_videos.append((video, f"Cost ${_cost:.4f} exceeds ceiling"))
            console.print(
                f"  [red]FAILED {video_id}: Estimated cost ${_cost:.4f} exceeds ceiling "
                f"${settings.caption_cost_ceiling:.2f}[/red]"
            )
            continue

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
                failed_count += 1
                console.print(
                    f"[yellow]Warning:[/yellow] Caption failed for chunk "
                    f"{chunk.chunk_index}: {exc}"
                )

        # Always write, even if some chunks failed
        writer.write(video_id, chunks)
        succeeded.append(video_id)

        summary = f"({fresh_count} new, {cached_count} cached)"
        if failed_count:
            summary = f"({fresh_count} new, {cached_count} cached, {failed_count} failed)"
        console.print(f"[green]Captioned:[/green] {video_id} {summary}")

        # Cumulative budget tracking (D-09)
        cumulative_spend += fresh_count * settings.caption_cost_per_chunk
        console.print(
            f"  Video {idx}/{len(videos)} done. "
            f"Spent: ${cumulative_spend:.4f} / ${settings.caption_cost_ceiling:.2f} budget"
        )

    # Final summary
    console.print(
        f"\n[green]Done:[/green] {len(succeeded)} succeeded, {len(failed_videos)} failed"
    )
    if failed_videos:
        for path, err in failed_videos:
            console.print(f"  [red]-[/red] {path}: {err}")
        raise typer.Exit(code=1)


def _print_results(results: list[dict], query: str) -> None:
    """Render search results as a Rich table."""
    table = Table(title=f"Results for: {query}")
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Video", width=20, no_wrap=True)
    table.add_column("Time", width=15)
    table.add_column("Score", width=8)
    table.add_column("Snippet", width=50)
    table.add_column("Reasoning", width=40)

    for rank, r in enumerate(results, start=1):
        start = f"{r['start_time']:.1f}s"
        end = f"{r['end_time']:.1f}s"
        time_range = f"{start} – {end}"
        snippet = r.get("combined_text", "")[:60]
        score = f"{r.get('rrf_score', 0):.4f}"
        reasoning = r.get("reasoning", "")[:50]
        table.add_row(str(rank), r["video_id"], time_range, score, snippet, reasoning)

    console.print(table)


if __name__ == "__main__":
    app()
