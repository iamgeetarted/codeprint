"""Command-line interface for codeprint."""

from __future__ import annotations

import argparse
import asyncio
import csv as _csv_mod
import io as _io
import json as _json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

from . import __version__
from .config import load_config as _load_config
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
_cfg = _load_config()

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


# ── Structured output formatters ──────────────────────────────────────────────

def _format_scan_output(metrics: list, root: Path, fmt: str) -> str:
    """Serialize scan metrics to json, csv, or markdown."""

    def rel(path: str) -> str:
        try:
            return str(Path(path).relative_to(root))
        except ValueError:
            return path

    if fmt == "json":
        data = [
            {
                "file": rel(m.path),
                "language": m.language,
                "loc": m.loc,
                "code_lines": m.code_lines,
                "blank_lines": m.blank_lines,
                "comment_lines": m.comment_lines,
                "functions": m.function_count,
                "classes": m.class_count,
                "imports": m.import_count,
                "todos": m.todo_count,
                "max_fn_loc": m.max_function_lines,
                "avg_fn_loc": m.avg_function_lines,
                "complexity": m.complexity_score,
            }
            for m in metrics
        ]
        return _json.dumps(data, indent=2)

    elif fmt == "csv":
        buf = _io.StringIO()
        writer = _csv_mod.writer(buf)
        writer.writerow([
            "file", "language", "loc", "code_lines", "blank_lines",
            "comment_lines", "functions", "classes", "imports",
            "todos", "max_fn_loc", "avg_fn_loc", "complexity"
        ])
        for m in metrics:
            writer.writerow([
                rel(m.path), m.language, m.loc, m.code_lines, m.blank_lines,
                m.comment_lines, m.function_count, m.class_count, m.import_count,
                m.todo_count, m.max_function_lines, m.avg_function_lines, m.complexity_score,
            ])
        return buf.getvalue()

    elif fmt == "markdown":
        lines = [
            f"# codeprint scan — {root}\n",
            f"| File | Lang | LOC | Fns | TODOs | Complexity |",
            f"|------|------|-----|-----|-------|------------|",
        ]
        for m in sorted(metrics, key=lambda x: -x.complexity_score)[:50]:
            lines.append(
                f"| `{rel(m.path)}` | {m.language} | {m.loc} | {m.function_count} | {m.todo_count} | {m.complexity_score} |"
            )
        return "\n".join(lines)

    return ""


def _format_dupes_output(pairs: list, root: Path, fmt: str) -> str:
    """Serialize dupe pairs to json, csv, or markdown."""

    def rel(path: str) -> str:
        try:
            return str(Path(path).relative_to(root))
        except ValueError:
            return path

    if fmt == "json":
        data = [{"file_a": rel(a), "file_b": rel(b), "similarity": round(sim, 4), "shared_tokens": tokens}
                for a, b, sim, tokens in pairs]
        return _json.dumps(data, indent=2)

    elif fmt == "csv":
        buf = _io.StringIO()
        writer = _csv_mod.writer(buf)
        writer.writerow(["file_a", "file_b", "similarity", "shared_tokens"])
        for a, b, sim, tokens in pairs:
            writer.writerow([rel(a), rel(b), round(sim, 4), tokens])
        return buf.getvalue()

    elif fmt == "markdown":
        lines = [
            f"# codeprint dupes — {root}\n",
            "| File A | File B | Similarity | Shared Tokens |",
            "|--------|--------|------------|---------------|",
        ]
        for a, b, sim, tokens in pairs:
            lines.append(f"| `{rel(a)}` | `{rel(b)}` | {sim:.1%} | {tokens} |")
        return "\n".join(lines)

    return ""


def _format_search_output(query: str, results: list, root: Path, fmt: str) -> str:
    """Serialize search results to json, csv, or markdown."""

    def rel(path: str) -> str:
        try:
            return str(Path(path).relative_to(root))
        except ValueError:
            return path

    if fmt == "json":
        data = [{"file": rel(path), "score": round(score, 4)} for path, score in results]
        return _json.dumps({"query": query, "results": data}, indent=2)

    elif fmt == "csv":
        buf = _io.StringIO()
        writer = _csv_mod.writer(buf)
        writer.writerow(["file", "score"])
        for path, score in results:
            writer.writerow([rel(path), round(score, 4)])
        return buf.getvalue()

    elif fmt == "markdown":
        lines = [
            f"# codeprint search — `{query}`\n",
            "| Score | File |",
            "|-------|------|",
        ]
        for path, score in results:
            lines.append(f"| {score:.1%} | `{rel(path)}` |")
        return "\n".join(lines)

    return ""


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_scan(args: argparse.Namespace) -> None:
    """Scan a directory and print the quality report."""
    root = _require_dir(args.path)
    metrics = asyncio.run(_scan_with_progress(root, quiet=args.quiet))

    if not metrics:
        console.print("[yellow]No supported source files found.[/yellow]")
        sys.exit(0)

    top = getattr(args, "top", None) or _cfg.get("defaults", {}).get("top", 15)
    fmt = getattr(args, "format", None) or _cfg.get("defaults", {}).get("format", "table")

    if fmt and fmt != "table":
        print(_format_scan_output(metrics, root, fmt))
        return

    render_summary(metrics, root)
    if not args.summary_only:
        render_top_files(metrics, root, n=top)
    if args.ai:
        stream_insights(metrics, root)


def cmd_dupes(args: argparse.Namespace) -> None:
    """Find near-duplicate files using TF-IDF cosine similarity."""
    root = _require_dir(args.path)
    metrics = asyncio.run(_scan_with_progress(root, quiet=args.quiet))

    if not metrics:
        console.print("[yellow]No source files found.[/yellow]")
        sys.exit(0)

    fmt = getattr(args, "format", None) or _cfg.get("defaults", {}).get("format", "table")
    threshold = getattr(args, "threshold", None) or _cfg.get("defaults", {}).get("threshold", 0.70)
    top = getattr(args, "top", None) or _cfg.get("defaults", {}).get("top", 20)

    console.print(
        f"[dim]Building TF-IDF vectors for {len(metrics)} files…[/dim]",
        file=sys.stderr if fmt and fmt != "table" else sys.stdout,
    )
    paths = [m.path for m in metrics]
    sources: list[str] = []
    for m in metrics:
        try:
            sources.append(Path(m.path).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            sources.append("")

    pairs = find_similar_pairs(paths, sources, threshold=threshold, top_k=top)

    if fmt and fmt != "table":
        print(_format_dupes_output(pairs, root, fmt))
        return

    render_duplicates(pairs, root)


def cmd_search(args: argparse.Namespace) -> None:
    """Semantically search the codebase for files matching a query."""
    root = _require_dir(args.path)
    query = " ".join(args.query)
    metrics = asyncio.run(_scan_with_progress(root, quiet=args.quiet))

    if not metrics:
        console.print("[yellow]No source files found.[/yellow]")
        sys.exit(0)

    top = getattr(args, "top", None) or _cfg.get("defaults", {}).get("top", 10)
    fmt = getattr(args, "format", None) or _cfg.get("defaults", {}).get("format", "table")

    console.print(f"[dim]Building TF-IDF index for {len(metrics)} files…[/dim]")
    paths = [m.path for m in metrics]
    sources: list[str] = []
    for m in metrics:
        try:
            sources.append(Path(m.path).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            sources.append("")

    results = semantic_search(query, paths, sources, top_k=top)

    if fmt and fmt != "table":
        print(_format_search_output(query, results, root, fmt))
        return

    render_search_results(query, results, root)


def cmd_insights(args: argparse.Namespace) -> None:
    """Stream an AI architectural summary of the codebase."""
    root = _require_dir(args.path)
    metrics = asyncio.run(_scan_with_progress(root, quiet=args.quiet))

    if not metrics:
        console.print("[yellow]No source files found.[/yellow]")
        sys.exit(0)

    stream_insights(metrics, root)


def cmd_diff(args: argparse.Namespace) -> None:
    """Compare quality metrics of two codebases side-by-side."""
    from rich.table import Table
    from rich.table import box as rbox
    from rich.panel import Panel

    root_a = _require_dir(args.path_a)
    root_b = _require_dir(args.path_b)

    console.print(f"[dim]Scanning A: {root_a}…[/dim]")
    metrics_a = asyncio.run(_scan_with_progress(root_a, quiet=True))
    console.print(f"[dim]Scanning B: {root_b}…[/dim]")
    metrics_b = asyncio.run(_scan_with_progress(root_b, quiet=True))

    def agg(metrics: list) -> dict:
        if not metrics:
            return {}
        return {
            "files": len(metrics),
            "loc": sum(m.loc for m in metrics),
            "code_lines": sum(m.code_lines for m in metrics),
            "functions": sum(m.function_count for m in metrics),
            "classes": sum(m.class_count for m in metrics),
            "todos": sum(m.todo_count for m in metrics),
            "avg_complexity": round(sum(m.complexity_score for m in metrics) / len(metrics), 1),
            "max_fn_loc": max((m.max_function_lines for m in metrics), default=0),
        }

    a = agg(metrics_a)
    b = agg(metrics_b)

    if not a or not b:
        console.print("[yellow]One or both directories have no supported files.[/yellow]")
        return

    label_a = root_a.name
    label_b = root_b.name

    table = Table(
        box=rbox.ROUNDED, show_header=True,
        header_style="bold cyan", border_style="dim",
    )
    table.add_column("Metric", style="dim")
    table.add_column(label_a, justify="right", style="cyan")
    table.add_column(label_b, justify="right", style="magenta")
    table.add_column("Δ", justify="right")

    def diff_row(label: str, key: str, lower_is_better: bool = False) -> None:
        va = a.get(key, 0)
        vb = b.get(key, 0)
        delta = vb - va
        if delta == 0:
            delta_str = "[dim]—[/dim]"
        elif (delta < 0) == lower_is_better:
            delta_str = f"[green]{delta:+.1f}[/green]" if isinstance(delta, float) else f"[green]{delta:+d}[/green]"
        else:
            delta_str = f"[red]{delta:+.1f}[/red]" if isinstance(delta, float) else f"[red]{delta:+d}[/red]"
        va_str = f"{va:.1f}" if isinstance(va, float) else str(va)
        vb_str = f"{vb:.1f}" if isinstance(vb, float) else str(vb)
        table.add_row(label, va_str, vb_str, delta_str)

    diff_row("Files", "files")
    diff_row("Total LOC", "loc")
    diff_row("Code lines", "code_lines")
    diff_row("Functions", "functions")
    diff_row("Classes", "classes")
    diff_row("Open TODOs", "todos", lower_is_better=True)
    diff_row("Avg complexity", "avg_complexity", lower_is_better=True)
    diff_row("Max fn LOC", "max_fn_loc", lower_is_better=True)

    console.print(Panel(table, title=f"Diff: {label_a} vs {label_b}", border_style="cyan", box=rbox.ROUNDED))
    console.print(
        f"\n  [dim]A:[/dim] {root_a}  [dim]B:[/dim] {root_b}"
        f"\n  [dim]Green Δ = improvement (B better than A) · Red Δ = regression[/dim]\n"
    )

    if args.ai:
        _stream_diff_insights(a, b, label_a, label_b, metrics_a, metrics_b)


def _stream_diff_insights(
    a: dict, b: dict, label_a: str, label_b: str,
    metrics_a: list, metrics_b: list,
) -> None:
    """Stream AI commentary on the diff between two codebases."""
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Set ANTHROPIC_API_KEY for AI diff insights.[/red]")
        return
    try:
        import anthropic
    except ImportError:
        console.print("[red]Install anthropic: pip install anthropic[/red]")
        return

    from rich.panel import Panel
    from rich import box

    prompt = (
        f"I compared two codebases:\n\n"
        f"**{label_a}**: {a['files']} files, {a['loc']:,} LOC, {a['functions']} functions, "
        f"{a['todos']} TODOs, avg complexity {a['avg_complexity']}, max fn LOC {a['max_fn_loc']}\n\n"
        f"**{label_b}**: {b['files']} files, {b['loc']:,} LOC, {b['functions']} functions, "
        f"{b['todos']} TODOs, avg complexity {b['avg_complexity']}, max fn LOC {b['max_fn_loc']}\n\n"
        "Please provide a concise comparison (3-5 sentences) covering:\n"
        "1. Which codebase looks healthier overall and why\n"
        "2. The most significant difference between them\n"
        "3. One concrete recommendation for the weaker codebase\n\n"
        "Be specific and developer-focused. Use markdown."
    )

    client = anthropic.Anthropic(api_key=api_key)
    console.print(Panel("[bold cyan]AI Diff Analysis[/bold cyan]", box=box.ROUNDED, border_style="cyan"))
    console.print()
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=384,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print("\n")


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
  codeprint scan . --format json        # structured JSON output
  codeprint dupes .                     # find near-duplicate files
  codeprint dupes . --threshold 0.8     # stricter similarity cutoff
  codeprint dupes . --format csv        # CSV output
  codeprint search . "database query logic"
  codeprint search . "auth middleware" --format markdown
  codeprint insights .                  # AI architectural analysis only
  codeprint diff ./v1 ./v2              # compare two codebases
  codeprint diff ./legacy ./rewrite --ai
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
    p_scan.add_argument("--format", "-f", choices=["table", "json", "csv", "markdown"],
                        default=None, help="Output format (default: table)")
    p_scan.set_defaults(func=cmd_scan)

    # dupes
    p_dupes = sub.add_parser("dupes", help="Find near-duplicate source files")
    p_dupes.add_argument("path", nargs="?", default=".", help="Directory to scan (default: .)")
    p_dupes.add_argument("--threshold", "-t", type=float, default=0.70, metavar="FLOAT",
                         help="Minimum cosine similarity to flag (0–1, default: 0.70)")
    p_dupes.add_argument("--top", type=int, default=20, metavar="N",
                         help="Maximum number of pairs to show (default: 20)")
    p_dupes.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p_dupes.add_argument("--format", "-f", choices=["table", "json", "csv", "markdown"],
                         default=None, help="Output format (default: table)")
    p_dupes.set_defaults(func=cmd_dupes)

    # search
    p_search = sub.add_parser("search", help="Semantically search for files matching a query")
    p_search.add_argument("path", help="Directory to scan")
    p_search.add_argument("query", nargs="+", help="Natural language query")
    p_search.add_argument("--top", type=int, default=10, metavar="N",
                          help="Number of results (default: 10)")
    p_search.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p_search.add_argument("--format", "-f", choices=["table", "json", "csv", "markdown"],
                          default=None, help="Output format (default: table)")
    p_search.set_defaults(func=cmd_search)

    # insights
    p_ins = sub.add_parser("insights", help="Stream an AI architectural summary (requires ANTHROPIC_API_KEY)")
    p_ins.add_argument("path", nargs="?", default=".", help="Directory to scan (default: .)")
    p_ins.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p_ins.set_defaults(func=cmd_insights)

    # diff
    p_diff = sub.add_parser("diff", help="Compare quality metrics of two directories side-by-side")
    p_diff.add_argument("path_a", help="First directory (baseline)")
    p_diff.add_argument("path_b", help="Second directory (compare against)")
    p_diff.add_argument("--ai", action="store_true",
                        help="Stream an AI commentary on the diff (requires ANTHROPIC_API_KEY)")
    p_diff.set_defaults(func=cmd_diff)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


def entry_point() -> None:
    main()
