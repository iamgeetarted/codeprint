"""Load user configuration from ~/.codeprint.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


_CONFIG_FILE = Path.home() / ".codeprint.toml"


def load_config() -> dict[str, Any]:
    """Return config from ~/.codeprint.toml; empty dict if absent."""
    try:
        with open(_CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        from rich.console import Console
        Console(stderr=True).print(f"[yellow]codeprint: config warning: {e}[/yellow]")
        return {}
