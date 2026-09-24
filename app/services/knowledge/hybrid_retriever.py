from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import numpy as np
from loguru import logger

from app.domain.evidence import KnowledgeChunk, RetrievalCandidate
from app.domain.knowledge_base import ChunkEmbedding
from app.services.knowledge.bm25_retriever import (
    DEFAULT_RETRIEVAL_POLICY_VERSION,
    BM25Index,
)
from app.services.knowledge.embedding_provider import (
    EmbeddingProvider,
    EmbeddingProviderError,
)

DEFAULT_HYBRID_POLICY_VERSION = "hybrid_rrf_v1"
DEFAULT_RRF_K = 60


class HybridRetriever:
    """Deterministic Hybrid Retrieval engine combining Okapi BM25 and exact Cosine Vector search with RRF."""

    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Sequence[ChunkEmbedding] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        retrieval_policy_version: str = DEFAULT_HYBRID_POLICY_VERSION,
        min_vector_similarity: float = 0.5,
    ) -> None:
        self.chunks = tuple(chunks)
        self.chunk_map: dict[str, KnowledgeChunk] = {c.chunk_id: c for c in self.chunks}
        self.embeddings = tuple(embeddings or ())
        self.embedding_map: dict[str, ChunkEmbedding] = {e.chunk_id: e for e in self.embeddings}
        self.embedding_provider = embedding_provider
        self.rrf_k = rrf_k
        self.retrieval_policy_version = retrieval_policy_version
        self.min_vector_similarity = min_vector_similarity

        # Pre-build lexical BM25 index
        self.bm25_index = BM25Index(
            self.chunks,
            retrieval_policy_version=retrieval_policy_version,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_scope_ids: Sequence[str] | None = None,
    ) -> tuple[list[RetrievalCandidate], str]:
        """Executes hybrid retrieval over scoped source documents.

        Returns:
            A tuple of (ranked_candidates, effective_retrieval_mode).
            effective_retrieval_mode will be one of:
                - "HYBRID"
                - "BM25_ONLY"
                - "BM25_FALLBACK"
                - "VECTOR_ONLY"
        """
        scopes_set = set(source_scope_ids) if source_scope_ids is not None else None

        # Filter candidate chunks by scopes
        scoped_chunks = [
            c for c in self.chunks
            if scopes_set is None or c.source_document_id in scopes_set
        ]

        if not scoped_chunks:
            return [], "HYBRID"

        # 1. Lexical BM25 Ranking
        # Request more candidates for better rank fusion overlap
        pool_k = max(top_k * 3, 50)
        bm25_candidates = self.bm25_index.search(
            query=query,
            top_k=pool_k,
            source_scope_ids=source_scope_ids,
        )

        # 2. Vector Semantic Ranking
        vector_candidates, vector_failed = self._vector_search(
            query=query,
            pool_k=pool_k,
            scopes_set=scopes_set,
        )

        # Fallback or degraded modes:
        if vector_failed:
            logger.warning("[HybridRetriever] Vector retrieval failed or unavailable; falling back to BM25.")
            final_cands = [
                RetrievalCandidate(
                    rank=idx + 1,
                    chunk_id=c.chunk_id,
                    source_document_id=c.source_document_id,
                    score=c.score,
                    retrieval_method="BM25_FALLBACK",
                    locator=c.locator,
                    excerpt=c.excerpt,
                )
                for idx, c in enumerate(bm25_candidates[:top_k])
            ]
            return final_cands, "BM25_FALLBACK"

        if not vector_candidates and bm25_candidates:
            # Vector had no matches / zero embeddings in scope
            final_cands = [
                RetrievalCandidate(
                    rank=idx + 1,
                    chunk_id=c.chunk_id,
                    source_document_id=c.source_document_id,
                    score=c.score,
                    retrieval_method="LEXICAL_BM25",
                    locator=c.locator,
                    excerpt=c.excerpt,
                )
                for idx, c in enumerate(bm25_candidates[:top_k])
            ]
            return final_cands, "BM25_ONLY"

        if vector_candidates and not bm25_candidates:
            final_cands = [
                RetrievalCandidate(
                    rank=idx + 1,
                    chunk_id=vc["chunk_id"],
                    source_document_id=vc["source_document_id"],
                    score=vc["score"],
                    retrieval_method="SEMANTIC_VECTOR",
                    locator=vc["locator"],
                    excerpt=vc["excerpt"],
                )
                for idx, vc in enumerate(vector_candidates[:top_k])
            ]
            return final_cands, "VECTOR_ONLY"

        # 3. Reciprocal Rank Fusion (RRF)
        # Map: chunk_id -> rrf_score
        rrf_scores: dict[str, float] = {}

        # Lexical ranks: 1-based
        for cand in bm25_candidates:
            cid = cand.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + cand.rank))

        # Vector ranks: 1-based
        for v_rank, vc in enumerate(vector_candidates, start=1):
            cid = vc["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + v_rank))

        # 4. Deterministic Tie-Breaking: Sort by (-rrf_score, chunk_id)
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (-rrf_scores[cid], cid),
        )

        fused_candidates: list[RetrievalCandidate] = []
        for rank_idx, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            chunk = self.chunk_map[cid]
            fused_candidates.append(
                RetrievalCandidate(
                    rank=rank_idx,
                    chunk_id=cid,
                    source_document_id=chunk.source_document_id,
                    score=round(rrf_scores[cid], 6),
                    retrieval_method="HYBRID_RRF",
                    locator=dict(chunk.locator or {}),
                    excerpt=chunk.normalized_text,
                )
            )

        return fused_candidates, "HYBRID"

    def _vector_search(
        self,
        query: str,
        pool_k: int,
        scopes_set: set[str] | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Computes exact cosine similarity over scoped chunk embeddings."""
        if not self.embedding_provider:
            return [], False

        # Filter available embeddings for scoped chunks
        valid_embs: list[tuple[KnowledgeChunk, ChunkEmbedding]] = []
        for cid, emb in self.embedding_map.items():
            chunk = self.chunk_map.get(cid)
            if chunk and (scopes_set is None or chunk.source_document_id in scopes_set):
                valid_embs.append((chunk, emb))

        if not valid_embs:
            return [], False

        try:
            q_vec = np.array(self.embedding_provider.embed_text(query), dtype=np.float32)
        except EmbeddingProviderError as exc:
            logger.warning(f"[HybridRetriever] Embedding query failed: {exc}")
            return [], True
        except Exception as exc:
            logger.error(f"[HybridRetriever] Unexpected error embedding query: {exc}")
            return [], True

        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-12:
            return [], False

        q_normed = q_vec / q_norm

        matrix = np.array([e[1].vector for e in valid_embs], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms < 1e-12, 1.0, norms)
        normed_matrix = matrix / norms

        sims = np.dot(normed_matrix, q_normed)

        scored: list[dict[str, Any]] = []
        for (chunk, _), sim in zip(valid_embs, sims):
            sim_val = float(sim)
            if sim_val >= self.min_vector_similarity:
                scored.append({
                    "chunk_id": chunk.chunk_id,
                    "source_document_id": chunk.source_document_id,
                    "score": sim_val,
                    "locator": dict(chunk.locator or {}),
                    "excerpt": chunk.normalized_text,
                })

        # Sort by (-score, chunk_id)
        scored.sort(key=lambda x: (-x["score"], x["chunk_id"]))
        return scored[:pool_k], False
