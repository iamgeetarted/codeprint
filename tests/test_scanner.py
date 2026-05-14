"""Tests for the async directory scanner."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from codeprint.scanner import scan_directory, scan_directory_sync, _collect_paths


_PY_FILE = """\
import os

class Watcher:
    def start(self):
        pass
"""

_JS_FILE = """\
const express = require('express');
function handler(req, res) { res.send('ok'); }
"""


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text(_PY_FILE)
    (tmp_path / "server.js").write_text(_JS_FILE)
    (tmp_path / "README.md").write_text("# docs")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cache.pyc").write_text("")
    return tmp_path


class TestCollectPaths:
    def test_finds_supported_extensions(self, tmp_path):
        _make_project(tmp_path)
        paths = _collect_paths(tmp_path)
        names = {p.name for p in paths}
        assert "main.py" in names
        assert "server.js" in names

    def test_skips_cache_dirs(self, tmp_path):
        _make_project(tmp_path)
        paths = _collect_paths(tmp_path)
        assert all("__pycache__" not in str(p) for p in paths)

    def test_ignores_unsupported_files(self, tmp_path):
        _make_project(tmp_path)
        paths = _collect_paths(tmp_path)
        assert not any(p.suffix == ".md" for p in paths)


class TestScanDirectory:
    @pytest.mark.asyncio
    async def test_returns_metrics_for_each_file(self, tmp_path):
        _make_project(tmp_path)
        results = await scan_directory(tmp_path)
        assert len(results) == 2
        langs = {r.language for r in results}
        assert "Python" in langs
        assert "JavaScript" in langs

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, tmp_path):
        _make_project(tmp_path)
        calls: list[tuple[int, int]] = []
        await scan_directory(tmp_path, progress_cb=lambda d, t: calls.append((d, t)))
        assert len(calls) == 2
        assert calls[-1][0] == calls[-1][1]  # final call: done == total

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path):
        results = await scan_directory(tmp_path)
        assert results == []

    def test_sync_wrapper(self, tmp_path):
        _make_project(tmp_path)
        results = scan_directory_sync(tmp_path)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_skips_node_modules(self, tmp_path):
        _make_project(tmp_path)
        nm = tmp_path / "node_modules" / "lib"
        nm.mkdir(parents=True)
        (nm / "helper.js").write_text("function x(){}")
        results = await scan_directory(tmp_path)
        # helper.js lives inside node_modules — it must not appear in results
        assert not any(r.path.endswith("helper.js") for r in results)
