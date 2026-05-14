"""Static code metrics extraction for Python and JavaScript files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileMetrics:
    """Quality metrics for a single source file."""

    path: str
    language: str
    loc: int = 0                  # total lines
    code_lines: int = 0           # non-blank, non-comment lines
    blank_lines: int = 0
    comment_lines: int = 0
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    todo_count: int = 0
    max_function_lines: int = 0
    avg_function_lines: float = 0.0
    complexity_score: float = 0.0  # heuristic: weighted sum of key metrics


# ── Per-language extractors ───────────────────────────────────────────────────

_TODO_RE = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|BUG)\b", re.IGNORECASE)
_PY_DEF_RE = re.compile(r"^\s*(async\s+)?def\s+\w+")
_PY_CLASS_RE = re.compile(r"^\s*class\s+\w+")
_PY_IMPORT_RE = re.compile(r"^\s*(import|from)\s+\w+")
_PY_COMMENT_RE = re.compile(r"^\s*#")

_JS_FUNC_RE = re.compile(
    r"\b(function\s+\w+\s*\(|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:\w+\s*)?\(|=>\s*\{)"
)
_JS_CLASS_RE = re.compile(r"\bclass\s+\w+")
_JS_IMPORT_RE = re.compile(r"^\s*(import\s+|require\s*\()")
_JS_COMMENT_RE = re.compile(r"^\s*(//|/\*|\*)")
_TODO_JS_RE = re.compile(r"//\s*(TODO|FIXME|HACK|XXX|BUG)\b", re.IGNORECASE)


def _extract_python(lines: list[str], fmetrics: FileMetrics) -> None:
    func_starts: list[int] = []
    func_lengths: list[int] = []
    current_func_start: int | None = None
    current_indent: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            fmetrics.blank_lines += 1
            continue

        if _PY_COMMENT_RE.match(line):
            fmetrics.comment_lines += 1
        else:
            fmetrics.code_lines += 1

        if _TODO_RE.search(line):
            fmetrics.todo_count += 1

        if _PY_IMPORT_RE.match(line):
            fmetrics.import_count += 1

        if _PY_CLASS_RE.match(line):
            fmetrics.class_count += 1

        if _PY_DEF_RE.match(line):
            # Close previous function span
            if current_func_start is not None:
                func_lengths.append(i - current_func_start)
            indent = len(line) - len(line.lstrip())
            current_func_start = i
            current_indent = indent
            fmetrics.function_count += 1

    if current_func_start is not None:
        func_lengths.append(len(lines) - current_func_start)

    if func_lengths:
        fmetrics.max_function_lines = max(func_lengths)
        fmetrics.avg_function_lines = round(sum(func_lengths) / len(func_lengths), 1)


def _extract_js(lines: list[str], fmetrics: FileMetrics) -> None:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            fmetrics.blank_lines += 1
            continue
        if _JS_COMMENT_RE.match(line):
            fmetrics.comment_lines += 1
        else:
            fmetrics.code_lines += 1

        if _TODO_JS_RE.search(line) or _TODO_RE.search(line):
            fmetrics.todo_count += 1
        if _JS_IMPORT_RE.match(line):
            fmetrics.import_count += 1
        if _JS_CLASS_RE.search(line):
            fmetrics.class_count += 1
        # Count function expressions / declarations on each line
        fmetrics.function_count += len(_JS_FUNC_RE.findall(line))


def extract_metrics(path: Path) -> FileMetrics | None:
    """Parse *path* and return a FileMetrics instance, or None on read error."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        lang = "Python"
    elif suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx"}:
        lang = "JavaScript"
    else:
        return None

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    lines = text.splitlines()
    fm = FileMetrics(path=str(path), language=lang, loc=len(lines))

    if lang == "Python":
        _extract_python(lines, fm)
    else:
        _extract_js(lines, fm)

    # Heuristic complexity: penalise large functions, many TODOs, dense imports
    fm.complexity_score = round(
        fm.max_function_lines * 0.15
        + fm.todo_count * 2.0
        + fm.function_count * 0.5
        + fm.import_count * 0.3,
        1,
    )

    return fm
