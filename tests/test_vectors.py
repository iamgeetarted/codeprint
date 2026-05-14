"""Tests for TF-IDF vector similarity."""

from __future__ import annotations

import numpy as np
import pytest

from codeprint.vectors import (
    build_tfidf_matrix,
    find_similar_pairs,
    semantic_search,
    tokenise,
)


class TestTokenise:
    def test_filters_short_tokens(self):
        tokens = tokenise("if is or")
        assert all(len(t) >= 3 for t in tokens)

    def test_filters_stop_words(self):
        tokens = tokenise("def class return import")
        assert tokens == []

    def test_extracts_identifiers(self):
        tokens = tokenise("def calculate_tax(income, rate):")
        assert "calculate_tax" in tokens or "calculate" in tokens
        assert "income" in tokens
        assert "rate" in tokens

    def test_lowercases(self):
        tokens = tokenise("MyClass MyMethod")
        assert all(t == t.lower() for t in tokens)


class TestTfidf:
    def test_matrix_shape(self):
        docs = ["alpha beta gamma", "delta epsilon zeta"]
        matrix, vocab = build_tfidf_matrix(docs)
        assert matrix.shape[0] == 2
        assert matrix.shape[1] == len(vocab)

    def test_normalised_rows(self):
        docs = ["foo bar baz", "qux quux corge"]
        matrix, _ = build_tfidf_matrix(docs)
        norms = np.linalg.norm(matrix, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_identical_docs_high_similarity(self):
        doc = "calculate_invoice total_amount apply_discount"
        matrix, _ = build_tfidf_matrix([doc, doc])
        sim = float(matrix[0] @ matrix[1])
        assert sim > 0.95

    def test_unrelated_docs_low_similarity(self):
        doc_a = "alpha beta gamma delta epsilon"
        doc_b = "zeta eta theta iota kappa"
        matrix, _ = build_tfidf_matrix([doc_a, doc_b])
        sim = float(matrix[0] @ matrix[1])
        assert sim < 0.3

    def test_empty_corpus(self):
        matrix, vocab = build_tfidf_matrix([])
        assert matrix.shape[0] == 0


class TestFindSimilarPairs:
    def test_identical_files_detected(self):
        source = "def fetch_data(url):\n    return requests.get(url).json()"
        pairs = find_similar_pairs(["a.py", "b.py"], [source, source], threshold=0.9)
        assert len(pairs) == 1
        assert pairs[0].similarity > 0.95

    def test_unrelated_files_not_flagged(self):
        a = "alpha beta gamma delta epsilon"
        b = "zeta eta theta iota kappa lambda"
        pairs = find_similar_pairs(["a.py", "b.py"], [a, b], threshold=0.7)
        assert len(pairs) == 0

    def test_sorted_by_similarity(self):
        sources = [
            "calculate_tax calculate_discount apply_tax",
            "calculate_tax calculate_discount apply_tax extra",
            "completely different words here nothing shared",
        ]
        paths = ["a.py", "b.py", "c.py"]
        pairs = find_similar_pairs(paths, sources, threshold=0.5)
        for i in range(len(pairs) - 1):
            assert pairs[i].similarity >= pairs[i + 1].similarity

    def test_returns_at_most_top_k(self):
        sources = [f"word_{i} token_{i} item_{i}" for i in range(10)]
        paths = [f"{i}.py" for i in range(10)]
        pairs = find_similar_pairs(paths, sources, threshold=0.0, top_k=3)
        assert len(pairs) <= 3


class TestSemanticSearch:
    def test_returns_most_relevant(self):
        paths = ["db.py", "views.py", "tests.py"]
        sources = [
            "database query connection pool execute transaction",
            "render template response context view",
            "assert test fixture mock setup teardown",
        ]
        results = semantic_search("sql database query", paths, sources, top_k=3)
        assert results[0][0] == "db.py"

    def test_empty_sources(self):
        results = semantic_search("query", [], [], top_k=5)
        assert results == []

    def test_respects_top_k(self):
        paths = [f"file{i}.py" for i in range(20)]
        sources = [f"word_{i} token_{i}" for i in range(20)]
        results = semantic_search("word token", paths, sources, top_k=5)
        assert len(results) <= 5
