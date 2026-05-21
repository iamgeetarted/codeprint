"""Save and compare codeprint metric snapshots for trend analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_SNAPSHOT_DIR = Path.home() / ".codeprint" / "snapshots"


def _ensure_dir() -> None:
    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def save_snapshot(metrics: list, root: Path, label: str | None = None) -> str:
    """Serialize metrics and save to ~/.codeprint/snapshots/. Returns snapshot ID."""
    _ensure_dir()
    ts = int(time.time())
    snapshot_id = f"{root.name}_{ts}"
    data: dict[str, Any] = {
        "id": snapshot_id,
        "root": str(root),
        "label": label or "",
        "timestamp": ts,
        "file_count": len(metrics),
        "totals": {
            "loc": sum(m.loc for m in metrics),
            "code_lines": sum(m.code_lines for m in metrics),
            "functions": sum(m.function_count for m in metrics),
            "classes": sum(m.class_count for m in metrics),
            "todos": sum(m.todo_count for m in metrics),
        },
        "avg_complexity": round(sum(m.complexity_score for m in metrics) / len(metrics), 2) if metrics else 0,
        "files": [
            {
                "path": m.path,
                "loc": m.loc,
                "functions": m.function_count,
                "complexity": m.complexity_score,
                "todos": m.todo_count,
            }
            for m in metrics
        ],
    }
    snap_path = _SNAPSHOT_DIR / f"{snapshot_id}.json"
    snap_path.write_text(json.dumps(data, indent=2))
    return snapshot_id


def list_snapshots(root_filter: str | None = None) -> list[dict[str, Any]]:
    """Return all saved snapshots, sorted by timestamp descending."""
    _ensure_dir()
    snaps = []
    for p in sorted(_SNAPSHOT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            if root_filter and root_filter not in data.get("root", ""):
                continue
            snaps.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return snaps


def load_snapshot(snapshot_id: str) -> dict[str, Any] | None:
    """Load a specific snapshot by ID."""
    p = _SNAPSHOT_DIR / f"{snapshot_id}.json"
    if not p.exists():
        # Try prefix match
        matches = list(_SNAPSHOT_DIR.glob(f"{snapshot_id}*.json"))
        if not matches:
            return None
        p = matches[0]
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return None
