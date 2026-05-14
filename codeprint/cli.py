"""Command-line interface for codeprint."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

from . import __version__
from .dashboard import console as dash_console
from .dashboard import (
    render_duplicates,
    render_search_results,
    render_summary,
    render_top_files,
)
from .metrics import FileMetrics
from .scanner import scan_directory
from .summarizer import stream_insights
from .vectors import find_similar_pairs, semantic_search

console = Console()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_dir(raw: str) -> Path:
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        console.print(f"[red]Not a directory: {p}[/red]")
        sys.exit(1)
    return p


async def _scan_with_progress(root: Path, quiet: bool) -> list[FileMetrics]:
    """Run the async scan, showing a live progress bar unless *quiet*."""
    if quiet:
        return await scan_directory(root)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[rate]} files/s"),
        console=console,
        transient=True,
    )
    task_id = progress.add_task("Scanning…", total=None, rate="—")
    t0 = time.monotonic()
    completed_ref = [0]

    def _cb(done: int, total: int) -> None:
        completed_ref[0] = done
        elapsed = time.monotonic() - t0
        rate = f"{done / elapsed:.0f}" if elapsed > 0.1 else "—"
        progress.update(task_id, completed=done, total=total, rate=rate)

    with progress:
        results = await scan_directory(root, progress_cb=_cb)

    elapsed = time.monotonic() - t0
    console.print(
        f"[dim]Scanned {len(results)} files in {elapsed:.2f}s[/dim]"
    )
    return results


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_scan(args: argparse.Namespace) -> None:
    """Scan a directory and print the quality report."""
    root = _require_dir(args.path)
    metrics = asyncio.run(_scan_with_progress(root, quiet=args.quiet))

    if not metrics:
        console.print("[yellow]No supported source files found.[/yellow]")
        sys.exit(0)

    render_summary(metrics, root)

    if not args.summary_only:
        render_top_files(metrics, root, n=args.top)

    if args.ai:
        stream_insights(metrics, root)


def cmd_dupes(args: argparse.Namespace) -> None:
    """Find near-duplicate files using TF-IDF cosine similarity."""
    root = _require_dir(args.path)
    metrics = asyncio.run(_scan_with_progress(root, quiet=args.quiet))

    if not metrics:
        console.print("[yellow]No source files found.[/yellow]")
        sys.exit(0)

    console.print(f"[dim]Building TF-IDF vectors for {len(metrics)} files…[/dim]")
    paths = [m.path for m in metrics]
    sources: list[str] = []
    for m in metrics:
        try:
            sources.append(Path(m.path).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            sources.append("")

    pairs = find_similar_pairs(paths, sources, threshold=args.threshold, top_k=args.top)
    render_duplicates(pairs, root)


def cmd_search(args: argparse.Namespace) -> None:
    """Semantically search the codebase for files matching a query."""
    root = _require_dir(args.path)
    query = " ".join(args.query)
    metrics = asyncio.run(_scan_with_progress(root, quiet=args.quiet))

    if not metrics:
        console.print("[yellow]No source files found.[/yellow]")
        sys.exit(0)

    console.print(f"[dim]Building TF-IDF index for {len(metrics)} files…[/dim]")
    paths = [m.path for m in metrics]
    sources: list[str] = []
    for m in metrics:
        try:
            sources.append(Path(m.path).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            sources.append("")

    results = semantic_search(query, paths, sources, top_k=args.top)
    render_search_results(query, results, root)


def cmd_insights(args: argparse.Namespace) -> None:
    """Stream an AI architectural summary of the codebase."""
    root = _require_dir(args.path)
    metrics = asyncio.run(_scan_with_progress(root, quiet=args.quiet))

    if not metrics:
        console.print("[yellow]No source files found.[/yellow]")
        sys.exit(0)

    stream_insights(metrics, root)


# ── Argument parser ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="codeprint",
        description="Async codebase quality scanner with semantic duplicate detection and AI insights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  codeprint scan .                      # scan current directory
  codeprint scan /path/to/project --ai  # scan + AI architectural summary
  codeprint scan . --top 20 --summary-only
  codeprint dupes .                     # find near-duplicate files
  codeprint dupes . --threshold 0.8     # stricter similarity cutoff
  codeprint search . "database query logic"
  codeprint insights .                  # AI architectural analysis only
""",
    )
    parser.add_argument("--version", action="version", version=f"codeprint {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # scan
    p_scan = sub.add_parser("scan", help="Scan a directory and show a quality report")
    p_scan.add_argument("path", nargs="?", default=".", help="Directory to scan (default: .)")
    p_scan.add_argument("--top", type=int, default=15, metavar="N",
                        help="Number of files to show per table (default: 15)")
    p_scan.add_argument("--summary-only", action="store_true",
                        help="Print only the high-level summary, skip detailed tables")
    p_scan.add_argument("--ai", action="store_true",
                        help="Stream an AI architectural summary after the report")
    p_scan.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p_scan.set_defaults(func=cmd_scan)

    # dupes
    p_dupes = sub.add_parser("dupes", help="Find near-duplicate source files")
    p_dupes.add_argument("path", nargs="?", default=".", help="Directory to scan (default: .)")
    p_dupes.add_argument("--threshold", "-t", type=float, default=0.70, metavar="FLOAT",
                         help="Minimum cosine similarity to flag (0–1, default: 0.70)")
    p_dupes.add_argument("--top", type=int, default=20, metavar="N",
                         help="Maximum number of pairs to show (default: 20)")
    p_dupes.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p_dupes.set_defaults(func=cmd_dupes)

    # search
    p_search = sub.add_parser("search", help="Semantically search for files matching a query")
    p_search.add_argument("path", help="Directory to scan")
    p_search.add_argument("query", nargs="+", help="Natural language query")
    p_search.add_argument("--top", type=int, default=10, metavar="N",
                          help="Number of results (default: 10)")
    p_search.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p_search.set_defaults(func=cmd_search)

    # insights
    p_ins = sub.add_parser("insights", help="Stream an AI architectural summary (requires ANTHROPIC_API_KEY)")
    p_ins.add_argument("path", nargs="?", default=".", help="Directory to scan (default: .)")
    p_ins.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p_ins.set_defaults(func=cmd_insights)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


def entry_point() -> None:
    main()
