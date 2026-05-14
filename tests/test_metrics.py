"""Tests for static metrics extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from codeprint.metrics import extract_metrics, FileMetrics


_SAMPLE_PYTHON = """\
import os
import sys
from pathlib import Path

# TODO: refactor this module

class Foo:
    def bar(self, x):
        return x + 1

    def baz(self):
        # FIXME: broken
        pass


def standalone(a, b):
    return a * b
"""

_SIMPLE_JS = """\
import React from 'react';
// TODO: cleanup

class Counter extends React.Component {
    render() {
        return <div>{this.state.count}</div>;
    }
}

const add = (a, b) => a + b;
function multiply(a, b) { return a * b; }
"""


def _write_tmp(content: str, suffix: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8")
    f.write(content)
    f.flush()
    f.close()
    return Path(f.name)


class TestPythonMetrics:
    def test_loc(self):
        p = _write_tmp(_SAMPLE_PYTHON, ".py")
        fm = extract_metrics(p)
        assert fm is not None
        assert fm.loc == len(_SAMPLE_PYTHON.splitlines())

    def test_function_count(self):
        p = _write_tmp(_SAMPLE_PYTHON, ".py")
        fm = extract_metrics(p)
        assert fm is not None
        assert fm.function_count == 3  # bar, baz, standalone

    def test_class_count(self):
        p = _write_tmp(_SAMPLE_PYTHON, ".py")
        fm = extract_metrics(p)
        assert fm is not None
        assert fm.class_count == 1

    def test_import_count(self):
        p = _write_tmp(_SAMPLE_PYTHON, ".py")
        fm = extract_metrics(p)
        assert fm is not None
        assert fm.import_count == 3

    def test_todo_count(self):
        p = _write_tmp(_SAMPLE_PYTHON, ".py")
        fm = extract_metrics(p)
        assert fm is not None
        assert fm.todo_count == 2

    def test_language(self):
        p = _write_tmp(_SAMPLE_PYTHON, ".py")
        fm = extract_metrics(p)
        assert fm is not None
        assert fm.language == "Python"

    def test_complexity_score_positive(self):
        p = _write_tmp(_SAMPLE_PYTHON, ".py")
        fm = extract_metrics(p)
        assert fm is not None
        assert fm.complexity_score > 0


class TestJavaScriptMetrics:
    def test_language(self):
        p = _write_tmp(_SIMPLE_JS, ".js")
        fm = extract_metrics(p)
        assert fm is not None
        assert fm.language == "JavaScript"

    def test_class_detected(self):
        p = _write_tmp(_SIMPLE_JS, ".js")
        fm = extract_metrics(p)
        assert fm is not None
        assert fm.class_count >= 1

    def test_todo_detected(self):
        p = _write_tmp(_SIMPLE_JS, ".js")
        fm = extract_metrics(p)
        assert fm is not None
        assert fm.todo_count >= 1

    def test_unsupported_extension_returns_none(self):
        p = _write_tmp("hello world", ".txt")
        assert extract_metrics(p) is None
