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


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch a directory for changes and auto-rescan, showing a live Rich dashboard."""
    import os
    import time as _time
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table, box as rbox
    from rich.align import Align
    from rich.text import Text

    root = _require_dir(args.path)
    poll_interval = args.interval

    def _collect_mtimes(directory: Path) -> dict[str, float]:
        mtimes: dict[str, float] = {}
        EXTS = {".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".rb", ".cs", ".swift"}
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix in EXTS:
                try:
                    mtimes[str(p)] = p.stat().st_mtime
                except OSError:
                    pass
        return mtimes

    def _build_layout(
        metrics: list,
        root: Path,
        scan_time: float,
        scan_count: int,
        changed_files: list[str],
    ) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="files", ratio=2),
            Layout(name="sidebar", ratio=1),
        )

        ts = _time.strftime("%H:%M:%S")
        layout["header"].update(
            Panel(
                Align.center(Text(f"codeprint watch — {root.name}  ·  scan #{scan_count}  ·  {ts}", style="bold cyan")),
                border_style="cyan",
            )
        )

        # Top files by complexity
        top = sorted(metrics, key=lambda m: -m.complexity_score)[:15]
        t = Table(box=rbox.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
        t.add_column("File", style="dim")
        t.add_column("LOC", justify="right", style="white", width=6)
        t.add_column("Fns", justify="right", style="cyan", width=5)
        t.add_column("Cmplx", justify="right", style="yellow", width=6)
        t.add_column("TODOs", justify="right", style="red", width=6)
        for m in top:
            try:
                rel = str(Path(m.path).relative_to(root))
            except ValueError:
                rel = m.path
            t.add_row(rel, str(m.loc), str(m.function_count), str(m.complexity_score), str(m.todo_count))
        layout["files"].update(
            Panel(t, title=f"Files by Complexity ({len(metrics)} total)", border_style="cyan")
        )

        # Summary sidebar
        if metrics:
            total_loc = sum(m.loc for m in metrics)
            total_fns = sum(m.function_count for m in metrics)
            total_todos = sum(m.todo_count for m in metrics)
            avg_cplx = sum(m.complexity_score for m in metrics) / len(metrics)
        else:
            total_loc = total_fns = total_todos = 0
            avg_cplx = 0.0

        s = Table(box=rbox.SIMPLE, show_header=False, expand=True)
        s.add_column("Key", style="cyan")
        s.add_column("Val", style="white", justify="right")
        s.add_row("Files", str(len(metrics)))
        s.add_row("Total LOC", str(total_loc))
        s.add_row("Functions", str(total_fns))
        s.add_row("Avg Complexity", f"{avg_cplx:.1f}")
        s.add_row("TODOs", str(total_todos))
        s.add_row("Scan time", f"{scan_time:.2f}s")
        if changed_files:
            s.add_section()
            s.add_row("[dim]Changed[/dim]", "")
            for f in changed_files[-5:]:
                try:
                    rel = str(Path(f).relative_to(root))
                except ValueError:
                    rel = f
                s.add_row(f"  [dim]{rel[:22]}[/dim]", "")
        layout["sidebar"].update(Panel(s, title="Summary", border_style="yellow"))

        layout["footer"].update(
            Panel(
                Align.center(Text(f"Polling every {poll_interval}s  ·  Ctrl+C to exit", style="dim")),
                border_style="dim",
            )
        )
        return layout

    console.print(f"[cyan]Watching[/cyan] {root}  [dim](poll every {poll_interval}s)[/dim]")
    console.print("[dim]Press Ctrl+C to exit.[/dim]\n")

    prev_mtimes: dict[str, float] = {}
    scan_count = 0
    metrics: list = []
    scan_time = 0.0
    changed_files: list[str] = []

    try:
        # Initial scan
        t0 = _time.monotonic()
        metrics = asyncio.run(scan_directory(root))
        scan_time = _time.monotonic() - t0
        scan_count = 1
        prev_mtimes = _collect_mtimes(root)

        with Live(
            _build_layout(metrics, root, scan_time, scan_count, []),
            refresh_per_second=0.5,
            screen=True,
        ) as live:
            while True:
                _time.sleep(poll_interval)
                cur_mtimes = _collect_mtimes(root)
                changed = [
                    f for f, mt in cur_mtimes.items()
                    if prev_mtimes.get(f) != mt
                ] + [f for f in prev_mtimes if f not in cur_mtimes]

                if changed:
                    prev_mtimes = cur_mtimes
                    changed_files = changed
                    t0 = _time.monotonic()
                    metrics = asyncio.run(scan_directory(root))
                    scan_time = _time.monotonic() - t0
                    scan_count += 1

                live.update(_build_layout(metrics, root, scan_time, scan_count, changed_files))
    except KeyboardInterrupt:
        console.print("[dim]Watch stopped.[/dim]")


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Manage metric snapshots: save, list, or compare."""
    from .snapshot import save_snapshot, list_snapshots, load_snapshot
    from rich.table import Table, box as rbox
    from rich.panel import Panel
    import time as _time
    from datetime import datetime
    from typing import Any

    subcmd = args.snapshot_cmd

    if subcmd == "save":
        root = _require_dir(args.path)
        console.print(f"[dim]Scanning {root}…[/dim]")
        metrics = asyncio.run(_scan_with_progress(root, quiet=args.quiet))
        if not metrics:
            console.print("[yellow]No files found.[/yellow]")
            return
        snap_id = save_snapshot(metrics, root, label=args.label)
        console.print(f"[green]✓ Snapshot saved:[/green] [cyan]{snap_id}[/cyan]")
        console.print(
            f"  [dim]{len(metrics)} files  ·  "
            f"{sum(m.loc for m in metrics):,} LOC  ·  "
            f"avg complexity {sum(m.complexity_score for m in metrics)/len(metrics):.1f}[/dim]"
        )

    elif subcmd == "list":
        snaps = list_snapshots()
        if not snaps:
            console.print("[dim]No snapshots saved yet. Run: codeprint snapshot save PATH[/dim]")
            return
        t = Table(box=rbox.ROUNDED, show_header=True, header_style="bold cyan", border_style="dim")
        t.add_column("ID", style="cyan")
        t.add_column("Root", style="dim")
        t.add_column("When", style="dim", width=18)
        t.add_column("Files", justify="right")
        t.add_column("LOC", justify="right")
        t.add_column("Avg Cplx", justify="right")
        t.add_column("Label", style="dim")
        for s in snaps:
            when = datetime.fromtimestamp(s["timestamp"]).strftime("%Y-%m-%d %H:%M")
            t.add_row(
                s["id"],
                Path(s["root"]).name,
                when,
                str(s["file_count"]),
                str(s["totals"]["loc"]),
                str(s["avg_complexity"]),
                s.get("label") or "",
            )
        console.print(Panel(t, title=f"Snapshots ({len(snaps)})", border_style="cyan", box=rbox.ROUNDED))

    elif subcmd == "compare":
        snap_a = load_snapshot(args.snap_a)
        snap_b = load_snapshot(args.snap_b)
        if not snap_a:
            console.print(f"[red]Snapshot not found: {args.snap_a}[/red]")
            return
        if not snap_b:
            console.print(f"[red]Snapshot not found: {args.snap_b}[/red]")
            return

        t = Table(box=rbox.ROUNDED, show_header=True, header_style="bold cyan", border_style="dim")
        t.add_column("Metric", style="dim")
        t.add_column(Path(snap_a["root"]).name + " (A)", justify="right", style="cyan")
        t.add_column(Path(snap_b["root"]).name + " (B)", justify="right", style="magenta")
        t.add_column("Δ", justify="right")

        def _row(label: str, key_path: list[str], lower_is_better: bool = False) -> None:
            va: Any = snap_a
            vb: Any = snap_b
            for k in key_path:
                va = va.get(k, 0)
                vb = vb.get(k, 0)
            delta = (vb or 0) - (va or 0)
            if delta == 0:
                ds = "[dim]—[/dim]"
            elif (delta < 0) == lower_is_better:
                ds = f"[green]{delta:+}[/green]"
            else:
                ds = f"[red]{delta:+}[/red]"
            t.add_row(label, str(va), str(vb), ds)

        _row("Files", ["file_count"])
        _row("Total LOC", ["totals", "loc"])
        _row("Functions", ["totals", "functions"])
        _row("TODOs", ["totals", "todos"], lower_is_better=True)
        _row("Avg Complexity", ["avg_complexity"], lower_is_better=True)

        ta = datetime.fromtimestamp(snap_a["timestamp"]).strftime("%Y-%m-%d %H:%M")
        tb = datetime.fromtimestamp(snap_b["timestamp"]).strftime("%Y-%m-%d %H:%M")
        title = f"Snapshot compare: {snap_a['id']} [{ta}] vs {snap_b['id']} [{tb}]"
        console.print(Panel(t, title=title, border_style="cyan", box=rbox.ROUNDED))

    else:
        console.print("[dim]Usage: codeprint snapshot save PATH | list | compare SNAP_A SNAP_B[/dim]")


def cmd_completions(args: argparse.Namespace) -> None:
    """Print shell completion script for bash, zsh, or fish."""
    shell = args.shell

    if shell == "bash":
        script = r"""# codeprint bash completion
# eval "$(codeprint completions bash)"
_codeprint_complete() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    local cmds="scan dupes search insights diff watch snapshot completions"
    local snap_cmds="save list compare"

    if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=($(compgen -W "$cmds" -- "$cur"))
    elif [[ $COMP_CWORD -eq 2 ]]; then
        case $prev in
            snapshot)    COMPREPLY=($(compgen -W "$snap_cmds" -- "$cur")) ;;
            completions) COMPREPLY=($(compgen -W "bash zsh fish" -- "$cur")) ;;
            scan|dupes|search|insights|watch) COMPREPLY=($(compgen -d -- "$cur")) ;;
        esac
    fi
}
complete -F _codeprint_complete codeprint
"""
    elif shell == "zsh":
        script = r"""#compdef codeprint
# eval "$(codeprint completions zsh)"
_codeprint() {
    local -a cmds
    cmds=(
        'scan:Scan a directory and show quality report'
        'dupes:Find near-duplicate source files'
        'search:Semantic code search'
        'insights:AI architectural summary'
        'diff:Compare two directories'
        'watch:Watch for changes with live dashboard'
        'snapshot:Save and compare metric snapshots'
        'completions:Print shell completion script'
    )
    _arguments -C '1:command:->cmd' '*::args:->args'
    case $state in
        cmd)  _describe 'commands' cmds ;;
        args)
            case $words[1] in
                snapshot)    local -a sc; sc=('save:Save snapshot' 'list:List snapshots' 'compare:Compare two snapshots'); _describe 'snapshot commands' sc ;;
                completions) _values 'shell' bash zsh fish ;;
                scan|dupes|search|insights|watch) _path_files -/ ;;
            esac ;;
    esac
}
_codeprint
"""
    elif shell == "fish":
        script = """# codeprint fish completion
# Save to ~/.config/fish/completions/codeprint.fish
set -l cmds scan dupes search insights diff watch snapshot completions
complete -c codeprint -f -n 'not __fish_seen_subcommand_from $cmds' -a "$cmds"
complete -c codeprint -n '__fish_seen_subcommand_from snapshot' -a 'save list compare'
complete -c codeprint -n '__fish_seen_subcommand_from completions' -a 'bash zsh fish'
complete -c codeprint -n '__fish_seen_subcommand_from scan dupes search insights watch' -a '(__fish_complete_directories)'
complete -c codeprint -n '__fish_seen_subcommand_from scan' -l ai -d 'Stream AI summary'
complete -c codeprint -n '__fish_seen_subcommand_from scan dupes search' -s f -l format -a 'table json csv markdown'
complete -c codeprint -n '__fish_seen_subcommand_from dupes' -s t -l threshold -d 'Similarity threshold (0-1)'
"""
    else:
        console.print(f"[red]Unknown shell '{shell}'. Choose: bash, zsh, fish[/red]")
        return

    print(script)


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
  codeprint watch .                     # live dashboard, auto-rescan on changes
  codeprint watch . --interval 5        # poll every 5 seconds
  codeprint snapshot save .             # save a metric snapshot
  codeprint snapshot save . --label "before refactor"
  codeprint snapshot list               # list all saved snapshots
  codeprint snapshot compare SNAP_A SNAP_B
  codeprint completions bash            # generate bash completion script
  eval "$(codeprint completions zsh)"   # activate zsh completions
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

    # watch
    p_watch = sub.add_parser("watch", help="Watch a directory for changes and auto-rescan with live dashboard")
    p_watch.add_argument("path", nargs="?", default=".", help="Directory to watch (default: .)")
    p_watch.add_argument("--interval", "-i", type=float, default=3.0, metavar="SECS",
                         help="Poll interval in seconds (default: 3)")
    p_watch.set_defaults(func=cmd_watch)

    # snapshot
    p_snap = sub.add_parser("snapshot", help="Save and compare metric snapshots over time")
    snap_sub = p_snap.add_subparsers(dest="snapshot_cmd")

    ps_save = snap_sub.add_parser("save", help="Save a metric snapshot of a directory")
    ps_save.add_argument("path", nargs="?", default=".", help="Directory to snapshot (default: .)")
    ps_save.add_argument("--label", "-l", metavar="TEXT", help="Optional label for this snapshot")
    ps_save.add_argument("--quiet", "-q", action="store_true")

    snap_sub.add_parser("list", help="List all saved snapshots")

    ps_cmp = snap_sub.add_parser("compare", help="Compare two snapshots side-by-side")
    ps_cmp.add_argument("snap_a", help="Snapshot ID (or prefix) for baseline")
    ps_cmp.add_argument("snap_b", help="Snapshot ID (or prefix) to compare against")

    p_snap.set_defaults(func=cmd_snapshot)

    # completions
    p_comp = sub.add_parser("completions", help="Print shell completion script (bash/zsh/fish)")
    p_comp.add_argument("shell", choices=["bash", "zsh", "fish"], help="Target shell")
    p_comp.set_defaults(func=cmd_completions)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


def entry_point() -> None:
    main()
