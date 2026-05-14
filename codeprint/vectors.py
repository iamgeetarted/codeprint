"""TF-IDF based semantic similarity for finding near-duplicate source files."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class SimilarPair:
    """Two files that share significant semantic overlap."""

    path_a: str
    path_b: str
    similarity: float   # cosine similarity in [0, 1]
    shared_tokens: list[str]


# ── Tokenisation ──────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z_]\w{2,}")   # identifiers / keywords ≥ 3 chars
_STOP_WORDS = frozenset(
    {
        "def", "class", "return", "import", "from", "for", "while", "if", "else",
        "elif", "with", "and", "not", "self", "true", "false", "none", "null",
        "const", "let", "var", "function", "async", "await", "yield", "lambda",
        "pass", "break", "continue", "try", "except", "finally", "raise", "in",
        "is", "or", "as", "global", "print", "str", "int", "bool", "list", "dict",
        "type", "the", "this", "that", "new", "new", "export", "default",
    }
)


def tokenise(source: str) -> list[str]:
    """Extract meaningful identifier tokens from source code."""
    raw = _TOKEN_RE.findall(source.lower())
    return [t for t in raw if t not in _STOP_WORDS]


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def build_tfidf_matrix(documents: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build a TF-IDF matrix for a list of text documents.

    Returns (matrix, vocabulary) where matrix.shape == (n_docs, vocab_size).
    """
    tokenised = [tokenise(doc) for doc in documents]

    # Build vocabulary
    all_tokens: set[str] = set()
    for tokens in tokenised:
        all_tokens.update(tokens)
    vocab = sorted(all_tokens)
    vocab_index = {t: i for i, t in enumerate(vocab)}
    n_docs = len(documents)
    v_size = len(vocab)

    if n_docs == 0 or v_size == 0:
        return np.zeros((n_docs, max(v_size, 1))), vocab

    # Term frequency per document (raw counts)
    tf = np.zeros((n_docs, v_size), dtype=np.float32)
    for doc_idx, tokens in enumerate(tokenised):
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        for token, cnt in counts.items():
            if token in vocab_index:
                tf[doc_idx, vocab_index[token]] = cnt / total

    # Inverse document frequency
    doc_freq = (tf > 0).sum(axis=0).astype(np.float32)
    idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1.0  # smoothed

    matrix = tf * idf

    # L2-normalise each row for cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32), vocab


def find_similar_pairs(
    paths: list[str],
    sources: list[str],
    threshold: float = 0.70,
    top_k: int = 20,
) -> list[SimilarPair]:
    """Return pairs of files whose TF-IDF cosine similarity exceeds *threshold*.

    Only the top *top_k* most similar pairs are returned.
    """
    if len(sources) < 2:
        return []

    matrix, vocab = build_tfidf_matrix(sources)
    # Cosine similarity = dot product (rows are already L2-normalised)
    sim_matrix = matrix @ matrix.T

    pairs: list[SimilarPair] = []
    n = len(sources)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                # Identify shared tokens
                tok_i = set(tokenise(sources[i]))
                tok_j = set(tokenise(sources[j]))
                shared = sorted(tok_i & tok_j)[:10]
                pairs.append(SimilarPair(paths[i], paths[j], round(sim, 3), shared))

    pairs.sort(key=lambda p: -p.similarity)
    return pairs[:top_k]


def semantic_search(
    query: str,
    paths: list[str],
    sources: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return the *top_k* files most semantically similar to *query*."""
    if not sources:
        return []
    all_docs = [query] + sources
    matrix, _ = build_tfidf_matrix(all_docs)
    query_vec = matrix[0]
    doc_vecs = matrix[1:]
    scores = doc_vecs @ query_vec
    ranked = sorted(enumerate(scores.tolist()), key=lambda x: -x[1])
    return [(paths[i], round(s, 3)) for i, s in ranked[:top_k] if s > 0]
