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
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from videosearch.captioner import GeminiCaptioner, QuotaExhaustedError
from videosearch.classifier import GeminiQueryClassifier
from videosearch.config import Settings
from videosearch.hybrid_retriever import HybridRetriever
from videosearch.index_builder import IndexBuilder
from videosearch.ingestion import IngestionPipeline
from videosearch.metadata_writer import MetadataWriter
from videosearch.reranker import ClaudeReranker

app = typer.Typer(name="videosearch", help="Search body-worn camera footage")
console = Console()

EVAL_QUERIES = [
    "Find all instances of a vehicle being pulled over at night",
    "Find every moment where someone raises their voice",
    "Locate all footage containing a person in a red shirt",
    "Find all interactions where an officer reads Miranda rights",
    "Find all license plates visible in the footage and tell me what each one says",
    "Find moments where a suspect is being handcuffed",
]


def _setting_float(settings: Settings, attr: str, default: float) -> float:
    """Read a numeric setting, falling back when mocks or invalid values leak in."""
    value = getattr(settings, attr, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


def _normalize_video_id(video: str) -> str:
    """Derive video_id from any path, stripping _720p suffix if present."""
    return Path(video).stem.replace("_720p", "")


def _is_cached(captioner: GeminiCaptioner, video_id: str, chunk_index: int) -> bool:
    """Treat only an explicit True cache hit as cached."""
    return captioner.is_cached(video_id, chunk_index) is True


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


@app.command("batch-eval")
def batch_eval(
    top_k: int = typer.Option(10, "--top-k", "-k", help="Results per query"),
) -> None:
    """Run all 6 example queries and display comparison table (INT-07)."""
    settings = Settings()
    classifier = GeminiQueryClassifier(settings)
    reranker = ClaudeReranker(settings)
    retriever = HybridRetriever(settings, classifier=classifier, reranker=reranker)

    table = Table(title="Batch Evaluation")
    table.add_column("Query", width=50)
    table.add_column("#Results", justify="right", width=9)
    table.add_column("Top Result", width=30)
    table.add_column("Top Snippet", width=50)

    for query in EVAL_QUERIES:
        try:
            results = retriever.retrieve(query, top_k=top_k)
        except Exception as e:
            table.add_row(query[:50], "ERROR", str(e)[:30], "")
            continue

        if not results:
            table.add_row(query[:50], "0", "-", "No results")
            continue

        top = results[0]
        count = str(len(results))
        top_result = f"{top['video_id']}  {top['start_time']:.1f}s-{top['end_time']:.1f}s"
        snippet = top.get("transcript_snippet", top.get("combined_text", ""))[:50]
        table.add_row(query[:50], count, top_result, snippet)

    console.print(table)


@app.command()
def estimate(
    video: str = typer.Argument(..., help="Path to video file"),
) -> None:
    """Show estimated captioning cost before any API call (ING-09)."""
    settings = Settings()
    video_id = _normalize_video_id(video)
    writer = MetadataWriter(metadata_dir=settings.metadata_dir)

    try:
        chunks = writer.load(video_id)
        chunk_count = len(chunks)
        # Count cached chunks using captioner's validity check (null/empty = miss)
        captioner = GeminiCaptioner(settings)
        cached_count = sum(
            1 for c in chunks
            if _is_cached(captioner, video_id, c.chunk_index)
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
        if result.returncode != 0:
            console.print(f"[red]Error:[/red] ffprobe failed for {video}. Cannot estimate cost.")
            raise typer.Exit(code=1)
        if not result.stdout.strip():
            console.print(f"[red]Error:[/red] ffprobe returned no duration for {video}.")
            raise typer.Exit(code=1)
        duration = float(result.stdout.strip())
        chunk_count = max(1, int(duration / 30))
        cached_count = 0

    new_chunks = chunk_count - cached_count
    cost_per_chunk = _setting_float(settings, "caption_cost_per_chunk", 0.003)
    total_cost = new_chunks * cost_per_chunk
    ceiling = _setting_float(settings, "caption_cost_ceiling", 5.0)

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
    cost_per_chunk = _setting_float(settings, "caption_cost_per_chunk", 0.003)
    per_video_ceiling = _setting_float(settings, "caption_cost_ceiling", 5.0)
    total_budget_ceiling = _setting_float(settings, "total_budget_ceiling", 30.0)

    # Batch cost estimation and confirmation (ING-09)
    total_new_chunks = 0
    for video in videos:
        video_id = _normalize_video_id(video)
        try:
            chunks = writer.load(video_id)
            total_new_chunks += sum(
                1 for c in chunks
                if not _is_cached(captioner, video_id, c.chunk_index)
            )
        except FileNotFoundError:
            pass  # will be caught in main loop

    estimated_total = total_new_chunks * cost_per_chunk
    if total_new_chunks > 0 and not yes:
        console.print(
            f"[bold]Estimated cost:[/bold] {total_new_chunks} new chunks "
            f"x ${cost_per_chunk:.4f} = ${estimated_total:.4f} "
            f"(ceiling: ${total_budget_ceiling:.2f})"
        )
        if not typer.confirm("Proceed with captioning?"):
            console.print("Aborted.")
            raise typer.Exit()

    succeeded: list[str] = []
    failed_videos: list[tuple[str, str]] = []
    cumulative_spend = 0.0

    for idx, video in enumerate(videos, 1):
        video_id = _normalize_video_id(video)

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

        # Cost preflight: enforce per-video and total ceilings before any API spend
        _new_count = sum(
            1 for c in chunks
            if not _is_cached(captioner, video_id, c.chunk_index)
        )
        _cost = _new_count * cost_per_chunk
        if _cost > per_video_ceiling:
            failed_videos.append((video, f"Cost ${_cost:.4f} exceeds per-video ceiling"))
            console.print(
                f"  [red]FAILED {video_id}: Estimated cost ${_cost:.4f} exceeds ceiling "
                f"${per_video_ceiling:.2f}[/red]"
            )
            continue

        # Enforce total budget ceiling BEFORE spending on this video
        if cumulative_spend + _cost > total_budget_ceiling:
            console.print(
                f"[red]Total budget ceiling ${total_budget_ceiling:.2f} would be exceeded "
                f"(spent ${cumulative_spend:.4f} + projected ${_cost:.4f}). "
                f"Stopping batch — remaining videos skipped.[/red]"
            )
            failed_videos.append((video, "Total budget ceiling would be exceeded"))
            for remaining_video in videos[idx:]:
                failed_videos.append((remaining_video, "Total budget ceiling reached"))
            break

        cached_count = 0
        fresh_count = 0
        failed_count = 0
        quota_exhausted = False

        for chunk in chunks:
            try:
                result = captioner.caption(compressed_path, chunk.start_time, chunk.end_time, chunk.chunk_index)
                chunk.visual_caption = result["caption"]
                if result.get("cached", False):
                    cached_count += 1
                else:
                    fresh_count += 1
            except QuotaExhaustedError:
                failed_count += 1
                quota_exhausted = True
                console.print(
                    f"[red]Quota exhausted[/red] at chunk {chunk.chunk_index}. "
                    f"Stopping — remaining chunks skipped."
                )
                failed_count += len(chunks) - (cached_count + fresh_count + failed_count)
                break
            except Exception as exc:
                failed_count += 1
                console.print(
                    f"[yellow]Warning:[/yellow] Caption failed for chunk "
                    f"{chunk.chunk_index}: {exc}"
                )

        # Write metadata with whatever captions we got
        writer.write(video_id, chunks)

        # Only count as succeeded if majority of chunks were captioned
        if failed_count > len(chunks) // 2:
            failed_videos.append((video, f"{failed_count}/{len(chunks)} chunks failed"))
        else:
            succeeded.append(video_id)

        summary = f"({fresh_count} new, {cached_count} cached)"
        if failed_count:
            color = "red" if failed_count > len(chunks) // 2 else "yellow"
            summary = f"({fresh_count} new, {cached_count} cached, {failed_count} failed)"
            if quota_exhausted:
                summary += " [quota exhausted]"
        console.print(f"[{color if failed_count else 'green'}]Captioned:[/{color if failed_count else 'green'}] {video_id} {summary}")

        # Cumulative budget tracking (D-09)
        cumulative_spend += fresh_count * cost_per_chunk
        console.print(
            f"  Video {idx}/{len(videos)} done. "
            f"Spent: ${cumulative_spend:.4f} / ${total_budget_ceiling:.2f} total budget"
        )

        # Enforce total budget ceiling across multi-video batch
        if cumulative_spend >= total_budget_ceiling:
            console.print(
                f"[red]Total budget ceiling ${total_budget_ceiling:.2f} reached. "
                f"Stopping batch — remaining videos skipped.[/red]"
            )
            for remaining_video in videos[idx:]:
                failed_videos.append((remaining_video, "Total budget ceiling reached"))
            break

    # Final summary
    console.print(
        f"\n[green]Done:[/green] {len(succeeded)} succeeded, {len(failed_videos)} failed"
    )
    if failed_videos:
        for path, err in failed_videos:
            console.print(f"  [red]-[/red] {path}: {err}")
        raise typer.Exit(code=1)


def _print_results(results: list[dict], query: str) -> None:
    """Render search results as Rich Panels (per D-01)."""
    console.print(f"\n[bold]Results for:[/bold] {query}\n")
    for rank, r in enumerate(results, 1):
        start = f"{r['start_time']:.1f}s"
        end = f"{r['end_time']:.1f}s"
        title = f"#{rank}  {r['video_id']}  {start} - {end}"

        score_line = f"[bold]Score:[/bold] {r.get('rrf_score', 0):.4f}"
        transcript = r.get("transcript_snippet", r.get("combined_text", ""))[:200]
        caption = r.get("visual_caption", "")[:200]
        reasoning = r.get("reasoning", "N/A")

        body = (
            f"{score_line}\n\n"
            f"[bold]Transcript:[/bold] {transcript}\n\n"
            f"[bold]Caption:[/bold] {caption}\n\n"
            f"[bold]Reasoning:[/bold] {reasoning}"
        )

        console.print(Panel(body, title=title, border_style="blue", padding=(0, 1)))


if __name__ == "__main__":
    app()
