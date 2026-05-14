"""Rich terminal output — scan reports, duplicate tables, and live progress."""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .metrics import FileMetrics
from .vectors import SimilarPair

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rel(path: str, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def _complexity_style(score: float) -> str:
    if score < 10:
        return "green"
    if score < 25:
        return "yellow"
    return "red"


def _loc_bar(loc: int, max_loc: int, width: int = 18) -> str:
    if max_loc == 0:
        return ""
    filled = max(1, round(loc / max_loc * width))
    return "█" * filled


# ── Scan report ──────────────────────────────────────────────────────────────

def render_summary(metrics: list[FileMetrics], root: Path) -> None:
    """Print a high-level summary panel."""
    if not metrics:
        console.print("[dim]No source files found.[/dim]")
        return

    total_loc = sum(m.loc for m in metrics)
    total_fns = sum(m.function_count for m in metrics)
    total_cls = sum(m.class_count for m in metrics)
    total_todos = sum(m.todo_count for m in metrics)
    total_imports = sum(m.import_count for m in metrics)
    py_files = sum(1 for m in metrics if m.language == "Python")
    js_files = len(metrics) - py_files

    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column("Key", style="cyan", width=18)
    t.add_column("Value", style="white")

    t.add_row("Files", str(len(metrics)))
    if py_files:
        t.add_row("  Python", str(py_files))
    if js_files:
        t.add_row("  JavaScript", str(js_files))
    t.add_row("Total LOC", f"{total_loc:,}")
    t.add_row("Functions", f"{total_fns:,}")
    t.add_row("Classes", f"{total_cls:,}")
    t.add_row("Imports", f"{total_imports:,}")
    t.add_row("TODOs / FIXMEs", str(total_todos))

    console.print(Panel(t, title=f"[bold cyan]codeprint[/bold cyan]  {root}", border_style="cyan", box=box.ROUNDED))


def render_top_files(metrics: list[FileMetrics], root: Path, n: int = 15) -> None:
    """Print the top-N files by LOC and by complexity score."""
    by_loc = sorted(metrics, key=lambda m: -m.loc)[:n]
    max_loc = by_loc[0].loc if by_loc else 1

    loc_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        pad_edge=False,
        expand=False,
    )
    loc_table.add_column("File", style="white")
    loc_table.add_column("Lang", style="dim", width=4)
    loc_table.add_column("LOC", justify="right", width=6)
    loc_table.add_column("Fns", justify="right", width=4)
    loc_table.add_column("TODOs", justify="right", width=6)
    loc_table.add_column("", width=20)  # bar

    for m in by_loc:
        loc_table.add_row(
            _rel(m.path, root),
            m.language[:2],
            str(m.loc),
            str(m.function_count),
            str(m.todo_count) if m.todo_count else "[dim]—[/dim]",
            f"[cyan]{_loc_bar(m.loc, max_loc)}[/cyan]",
        )
    console.print(Panel(loc_table, title=f"Largest Files (top {n})", border_style="dim", box=box.ROUNDED))

    # Complexity table
    by_complexity = sorted(metrics, key=lambda m: -m.complexity_score)[:n]

    cx_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold yellow",
        pad_edge=False,
        expand=False,
    )
    cx_table.add_column("File", style="white")
    cx_table.add_column("Score", justify="right", width=6)
    cx_table.add_column("Max fn LOC", justify="right", width=10)
    cx_table.add_column("Avg fn LOC", justify="right", width=10)
    cx_table.add_column("TODOs", justify="right", width=6)

    for m in by_complexity:
        style = _complexity_style(m.complexity_score)
        cx_table.add_row(
            _rel(m.path, root),
            Text(str(m.complexity_score), style=style),
            str(m.max_function_lines) if m.max_function_lines else "—",
            str(m.avg_function_lines) if m.avg_function_lines else "—",
            str(m.todo_count) if m.todo_count else "[dim]—[/dim]",
        )
    console.print(Panel(cx_table, title=f"Complexity Hotspots (top {n})", border_style="yellow", box=box.ROUNDED))


# ── Duplicate report ─────────────────────────────────────────────────────────

def render_duplicates(pairs: list[SimilarPair], root: Path) -> None:
    """Print a table of near-duplicate file pairs."""
    if not pairs:
        console.print("[green]✓ No near-duplicate files detected.[/green]")
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="magenta",
        expand=False,
    )
    table.add_column("File A", style="white")
    table.add_column("File B", style="white")
    table.add_column("Sim %", justify="right", width=7)
    table.add_column("Shared tokens", style="dim")

    for p in pairs:
        sim_pct = f"{p.similarity * 100:.0f}%"
        style = "red" if p.similarity >= 0.85 else "yellow"
        table.add_row(
            _rel(p.path_a, root),
            _rel(p.path_b, root),
            Text(sim_pct, style=style),
            ", ".join(p.shared_tokens[:8]),
        )

    console.print(
        Panel(
            table,
            title=f"Near-Duplicate Files — {len(pairs)} pair{'s' if len(pairs) != 1 else ''}",
            border_style="magenta",
            box=box.ROUNDED,
        )
    )
    console.print("[dim]  Similarity ≥85% likely duplication · 70–84% significant overlap[/dim]")


# ── Search results ────────────────────────────────────────────────────────────

def render_search_results(query: str, results: list[tuple[str, float]], root: Path) -> None:
    """Print semantic search results."""
    if not results:
        console.print(f"[dim]No results for '{query}'.[/dim]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", pad_edge=False)
    table.add_column("Score", justify="right", width=7)
    table.add_column("File", style="white")

    for path, score in results:
        pct = f"{score * 100:.0f}%"
        style = "green" if score >= 0.4 else ("yellow" if score >= 0.2 else "dim")
        table.add_row(Text(pct, style=style), _rel(path, root))

    console.print(Panel(table, title=f'Search: "{query}"', border_style="cyan", box=box.ROUNDED))
