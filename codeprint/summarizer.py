"""AI-powered codebase insights using streaming Claude API."""

from __future__ import annotations

import os
from pathlib import Path

from .metrics import FileMetrics


def _build_prompt(metrics: list[FileMetrics], root: Path) -> str:
    total_loc = sum(m.loc for m in metrics)
    total_fns = sum(m.function_count for m in metrics)
    total_todos = sum(m.todo_count for m in metrics)

    top_complex = sorted(metrics, key=lambda m: -m.complexity_score)[:8]
    top_loc = sorted(metrics, key=lambda m: -m.loc)[:8]
    langs = {}
    for m in metrics:
        langs[m.language] = langs.get(m.language, 0) + 1

    def rel(path: str) -> str:
        try:
            return str(Path(path).relative_to(root))
        except ValueError:
            return path

    complex_lines = "\n".join(
        f"  - {rel(m.path)}: score={m.complexity_score}, fns={m.function_count}, "
        f"max_fn_loc={m.max_function_lines}, todos={m.todo_count}"
        for m in top_complex
    )
    large_lines = "\n".join(
        f"  - {rel(m.path)}: {m.loc} LOC, {m.function_count} functions"
        for m in top_loc
    )

    return (
        f"I've scanned a codebase at '{root}' and collected these metrics:\n\n"
        f"**Overview**\n"
        f"- Files: {len(metrics)}\n"
        f"- Languages: {', '.join(f'{lang} ({n})' for lang, n in langs.items())}\n"
        f"- Total LOC: {total_loc:,}\n"
        f"- Functions: {total_fns:,}\n"
        f"- Open TODOs/FIXMEs: {total_todos}\n\n"
        f"**Complexity hotspots** (heuristic score = weighted fn size + todos + imports):\n"
        f"{complex_lines}\n\n"
        f"**Largest files**:\n"
        f"{large_lines}\n\n"
        "Please provide a concise architectural assessment (4-6 sentences) covering:\n"
        "1. What this codebase is probably doing (infer from file names/metrics)\n"
        "2. The top 2-3 quality concerns (large functions, many TODOs, size imbalance)\n"
        "3. One concrete refactoring recommendation\n\n"
        "Be specific, developer-focused, and practical. Use markdown."
    )


def stream_insights(metrics: list[FileMetrics], root: Path) -> None:
    """Stream an AI architectural summary to stdout using Claude Haiku."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        from rich.console import Console
        Console().print(
            "[red]ANTHROPIC_API_KEY not set.[/red]\n"
            "  Export it: [bold]export ANTHROPIC_API_KEY=your-key[/bold]"
        )
        return

    try:
        import anthropic
    except ImportError:
        from rich.console import Console
        Console().print("[red]Install anthropic: pip install anthropic[/red]")
        return

    from rich.console import Console
    from rich import box
    from rich.panel import Panel

    console = Console()
    prompt = _build_prompt(metrics, root)
    client = anthropic.Anthropic(api_key=api_key)

    console.print(Panel("[bold cyan]AI Architectural Insights[/bold cyan]", box=box.ROUNDED, border_style="cyan"))
    console.print()

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print("\n")
