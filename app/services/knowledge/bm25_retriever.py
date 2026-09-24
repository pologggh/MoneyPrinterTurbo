from __future__ import annotations

from collections import Counter
import math
import re
from typing import Sequence

from app.domain.evidence import (
    KnowledgeChunk,
    RetrievalCandidate,
)

DEFAULT_RETRIEVAL_POLICY_VERSION = "lexical_bm25_v1"


def tokenize(text: str) -> list[str]:
    """Deterministic multilingual tokenizer supporting English words and CJK characters."""
    if not text:
        return []
    # Lowercase
    clean = text.lower()
    # Extract Latin words/numbers and CJK characters
    # CJK unigrams: [\u4e00-\u9fa5]
    tokens: list[str] = []
    # Find word tokens (sequences of alphanumeric) and individual CJK characters
    pattern = re.compile(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")
    for match in pattern.finditer(clean):
        tokens.append(match.group(0))
    return tokens


class BM25Index:
    """In-memory Okapi BM25 index over a collection of KnowledgeChunks."""

    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
        k1: float = 1.5,
        b: float = 0.75,
        retrieval_policy_version: str = DEFAULT_RETRIEVAL_POLICY_VERSION,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.retrieval_policy_version = retrieval_policy_version
        self.chunks = list(chunks)
        self.doc_count = len(self.chunks)

        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.term_doc_freq: dict[str, int] = {}
        self.doc_freqs: list[Counter[str]] = []

        total_len = 0
        for chunk in self.chunks:
            tokens = tokenize(chunk.normalized_text)
            self.doc_tokens.append(tokens)
            d_len = len(tokens)
            self.doc_len.append(d_len)
            total_len += d_len

            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            for term in freq:
                self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1

        self.avg_doc_len = (total_len / self.doc_count) if self.doc_count > 0 else 0.0

    def idf(self, term: str) -> float:
        """Calculates standard Robertson-Spärck Jones BM25 IDF with smoothing to ensure non-negative values."""
        df = self.term_doc_freq.get(term, 0)
        # Standard non-negative BM25 IDF formula: ln(1 + (N - df + 0.5) / (df + 0.5))
        return math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_scope_ids: Sequence[str] | None = None,
    ) -> list[RetrievalCandidate]:
        """Searches indexed chunks using Okapi BM25 with deterministic tie-breaking."""
        query_tokens = tokenize(query)
        if not query_tokens or self.doc_count == 0:
            return []

        allowed_sources = set(source_scope_ids) if source_scope_ids is not None else None

        scored_candidates: list[tuple[float, str, int]] = []  # (score, chunk_id, doc_index)

        for doc_idx, chunk in enumerate(self.chunks):
            if allowed_sources is not None and chunk.source_document_id not in allowed_sources:
                continue

            doc_freq = self.doc_freqs[doc_idx]
            d_len = self.doc_len[doc_idx]
            score = 0.0

            for q_term in query_tokens:
                tf = doc_freq.get(q_term, 0)
                if tf == 0:
                    continue

                idf_val = self.idf(q_term)
                # BM25 term weight
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len)) if self.avg_doc_len > 0 else (tf + self.k1)
                score += idf_val * (numerator / denominator)

            if score > 0.0:
                scored_candidates.append((score, chunk.chunk_id, doc_idx))

        # Deterministic sorting: highest score first, tie-break on chunk_id ascending
        scored_candidates.sort(key=lambda x: (-x[0], x[1]))

        candidates: list[RetrievalCandidate] = []
        for rank, (score, _, doc_idx) in enumerate(scored_candidates[:top_k], start=1):
            chunk = self.chunks[doc_idx]
            excerpt = chunk.normalized_text[:200]
            candidates.append(
                RetrievalCandidate(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    source_document_id=chunk.source_document_id,
                    score=round(score, 4),
                    retrieval_method="LEXICAL_BM25",
                    locator=chunk.locator,
                    excerpt=excerpt,
                )
            )

        return candidates
