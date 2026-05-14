"""Async directory scanner — processes files concurrently via asyncio.TaskGroup."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

from .metrics import FileMetrics, extract_metrics

_SUPPORTED_EXTENSIONS = frozenset(
    {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx"}
)
_SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".ruff_cache",
        "node_modules", ".venv", "venv", "env", ".env",
        "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    }
)


def _collect_paths(root: Path, extensions: frozenset[str] | None = None) -> list[Path]:
    """Walk *root* and return all source files, skipping common non-source dirs."""
    exts = extensions or _SUPPORTED_EXTENSIONS
    results: list[Path] = []
    for p in root.rglob("*"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in exts:
            results.append(p)
    return results


async def _scan_file(path: Path) -> FileMetrics | None:
    """Run metrics extraction in a thread so it never blocks the event loop."""
    return await asyncio.to_thread(extract_metrics, path)


async def scan_directory(
    root: Path,
    extensions: frozenset[str] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    max_concurrent: int = 32,
) -> list[FileMetrics]:
    """Scan *root* concurrently and return FileMetrics for every source file.

    Args:
        root: Directory to scan.
        extensions: File extensions to include (default: .py, .js, .ts, …).
        progress_cb: Optional callback called with (completed, total) after each file.
        max_concurrent: Max simultaneous file-read tasks (semaphore).

    Returns:
        List of FileMetrics, one per successfully parsed file.
    """
    paths = _collect_paths(root, extensions)
    total = len(paths)
    completed = 0
    results: list[FileMetrics] = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(path: Path) -> FileMetrics | None:
        nonlocal completed
        async with semaphore:
            result = await _scan_file(path)
            completed += 1
            if progress_cb:
                progress_cb(completed, total)
            return result

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(_bounded(p)) for p in paths]

    for task in tasks:
        fm = task.result()
        if fm is not None:
            results.append(fm)

    return results


def scan_directory_sync(root: Path, **kwargs) -> list[FileMetrics]:
    """Synchronous wrapper around scan_directory for non-async callers."""
    return asyncio.run(scan_directory(root, **kwargs))
