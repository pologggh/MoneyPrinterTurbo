from __future__ import annotations

from collections.abc import Sequence
import json
import math
from pathlib import Path
import time
from typing import Any
import numpy as np
from sqlalchemy.orm import Session

from app.application.knowledge_retrieval_service import KnowledgeRetrievalService
from app.domain.retrieval_benchmark import (
    RELEVANT_GRADE_THRESHOLD,
    VALID_RELEVANCE_GRADES,
    ModeBenchmarkSummary,
    PerQueryBenchmarkResult,
    PerQueryRankedItem,
    QueryCategory,
    QueryDifficulty,
    RetrievalBenchmarkReport,
    RetrievalGoldenDataset,
    RetrievalGoldenQuery,
)
from app.services.knowledge.embedding_provider import EmbeddingProvider


# =============================================================================
# Metrics Calculator
# =============================================================================


class RetrievalMetricsCalculator:
    """Standard Information Retrieval (IR) metrics calculator for benchmark queries."""

    @staticmethod
    def compute_recall_at_k(
        retrieved_chunk_ids: Sequence[str],
        relevance_map: dict[str, int],
        k: int,
    ) -> float:
        """Recall@K = (number of relevant chunks retrieved in top K) / (total relevant chunks in ground truth)."""
        if k <= 0:
            return 0.0
        relevant_chunks = {cid for cid, grade in relevance_map.items() if grade >= RELEVANT_GRADE_THRESHOLD}
        if not relevant_chunks:
            return 0.0

        top_k_retrieved = retrieved_chunk_ids[:k]
        hits = len(set(top_k_retrieved) & relevant_chunks)
        return float(hits) / float(len(relevant_chunks))

    @staticmethod
    def compute_precision_at_k(
        retrieved_chunk_ids: Sequence[str],
        relevance_map: dict[str, int],
        k: int = 5,
    ) -> float:
        """Precision@K = (number of relevant chunks retrieved in top K) / K."""
        if k <= 0:
            return 0.0
        relevant_chunks = {cid for cid, grade in relevance_map.items() if grade >= RELEVANT_GRADE_THRESHOLD}
        if not relevant_chunks:
            return 0.0

        top_k_retrieved = retrieved_chunk_ids[:k]
        hits = len(set(top_k_retrieved) & relevant_chunks)
        return float(hits) / float(k)

    @staticmethod
    def compute_mrr(
        retrieved_chunk_ids: Sequence[str],
        relevance_map: dict[str, int],
    ) -> float:
        """MRR (Reciprocal Rank) = 1.0 / rank of first relevant chunk (1-based), or 0.0 if no hits."""
        relevant_chunks = {cid for cid, grade in relevance_map.items() if grade >= RELEVANT_GRADE_THRESHOLD}
        if not relevant_chunks:
            return 0.0

        for idx, cid in enumerate(retrieved_chunk_ids, start=1):
            if cid in relevant_chunks:
                return 1.0 / float(idx)
        return 0.0

    @staticmethod
    def compute_ndcg_at_k(
        retrieved_chunk_ids: Sequence[str],
        relevance_map: dict[str, int],
        k: int = 5,
    ) -> float:
        """Graded nDCG@K using exponential relevance gain (2^rel - 1) / log2(rank + 1)."""
        if k <= 0:
            return 0.0

        # DCG@K calculation
        dcg = 0.0
        for idx, cid in enumerate(retrieved_chunk_ids[:k], start=1):
            rel = relevance_map.get(cid, 0)
            if rel > 0:
                dcg += (math.pow(2, rel) - 1.0) / math.log2(idx + 1)

        # Ideal DCG@K (IDCG@K) calculation from ground truth sorted descending
        ideal_grades = sorted([g for g in relevance_map.values() if g > 0], reverse=True)[:k]
        if not ideal_grades:
            return 0.0

        idcg = 0.0
        for idx, rel in enumerate(ideal_grades, start=1):
            idcg += (math.pow(2, rel) - 1.0) / math.log2(idx + 1)

        if idcg <= 1e-12:
            return 0.0

        return dcg / idcg

    @staticmethod
    def compute_latency_percentiles(latencies_ms: Sequence[float]) -> tuple[float, float]:
        """Calculates (mean_latency_ms, p95_latency_ms)."""
        if not latencies_ms:
            return 0.0, 0.0
        arr = np.array(latencies_ms, dtype=np.float64)
        mean_val = float(np.mean(arr))
        p95_val = float(np.percentile(arr, 95))
        return round(mean_val, 2), round(p95_val, 2)

    @staticmethod
    def identify_failure_types(
        ranked_chunk_ids: Sequence[str],
        relevance_map: dict[str, int],
        hard_negatives: Sequence[str] = (),
        mode: str = "",
        bm25_hit_top5: bool | None = None,
        dense_hit_top5: bool | None = None,
        all_retrieved_chunk_ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Identifies diagnostic failure modes for failure analysis."""
        failures: list[str] = []
        relevant_chunks = {cid for cid, grade in relevance_map.items() if grade >= RELEVANT_GRADE_THRESHOLD}

        if relevant_chunks:
            top_5 = ranked_chunk_ids[:5]
            hits_in_top5 = set(top_5) & relevant_chunks

            if not hits_in_top5:
                failures.append("ZERO_RECALL_AT_5")

                # Check if relevant item exists beyond top-5
                pool = all_retrieved_chunk_ids if all_retrieved_chunk_ids is not None else ranked_chunk_ids
                hits_in_pool = set(pool) & relevant_chunks
                if hits_in_pool and not hits_in_top5:
                    failures.append("FIRST_HIT_BEYOND_TOP_5")

            # Check if any hard negative ranked in Top 5
            hn_set = set(hard_negatives)
            if hn_set and any(cid in hn_set for cid in top_5):
                failures.append("HARD_NEGATIVE_IN_TOP_5")

            # Cross-mode failure comparisons
            curr_mode = mode.upper()
            if curr_mode == "DENSE_ONLY":
                if not hits_in_top5 and bm25_hit_top5 is True:
                    failures.append("BM25_HIT_DENSE_MISS")
            elif curr_mode == "BM25_ONLY":
                if not hits_in_top5 and dense_hit_top5 is True:
                    failures.append("DENSE_HIT_BM25_MISS")

        return failures


# =============================================================================
# Dataset Validator & Loader
# =============================================================================


class RetrievalDatasetValidator:
    """Validates Golden Dataset structure, schema invariants, and optional corpus consistency."""

    @staticmethod
    def validate_dict(data: dict[str, Any]) -> RetrievalGoldenDataset:
        """Parses and validates a raw dict into a RetrievalGoldenDataset."""
        if not isinstance(data, dict):
            raise ValueError("Dataset payload must be a JSON object / dict.")

        dataset_name = data.get("dataset_name")
        if not dataset_name or not str(dataset_name).strip():
            raise ValueError("dataset_name cannot be empty.")

        raw_queries = data.get("queries")
        if not isinstance(raw_queries, (list, tuple)):
            raise ValueError("Dataset 'queries' field must be a list.")

        parsed_queries: list[RetrievalGoldenQuery] = []
        seen_query_ids: set[str] = set()

        for idx, q_data in enumerate(raw_queries):
            if not isinstance(q_data, dict):
                raise ValueError(f"Query at index {idx} must be an object.")

            qid = str(q_data.get("query_id") or "").strip()
            if not qid:
                raise ValueError(f"Query at index {idx} has missing or empty 'query_id'.")
            if qid in seen_query_ids:
                raise ValueError(f"Duplicate query_id '{qid}' detected.")
            seen_query_ids.add(qid)

            q_text = str(q_data.get("query") or "").strip()
            if not q_text:
                raise ValueError(f"Query '{qid}' has missing or empty 'query' text.")

            raw_rel = q_data.get("relevance", {})
            if not isinstance(raw_rel, dict):
                raise ValueError(f"Query '{qid}' relevance field must be a dictionary.")

            clean_rel: dict[str, int] = {}
            for cid, grade in raw_rel.items():
                if not isinstance(grade, int):
                    raise ValueError(f"Query '{qid}' relevance grade for chunk '{cid}' must be an integer.")
                if grade not in VALID_RELEVANCE_GRADES:
                    raise ValueError(
                        f"Query '{qid}' has invalid relevance grade {grade} for chunk '{cid}'. "
                        f"Must be one of {sorted(VALID_RELEVANCE_GRADES)}."
                    )
                clean_rel[str(cid)] = grade

            raw_diff = str(q_data.get("difficulty", "MEDIUM")).upper()
            try:
                diff_enum = QueryDifficulty(raw_diff)
            except ValueError:
                raise ValueError(
                    f"Query '{qid}' has invalid difficulty '{raw_diff}'. "
                    f"Allowed: {[d.value for d in QueryDifficulty]}."
                )

            raw_cat = str(q_data.get("category", "FACT")).upper()
            try:
                cat_enum = QueryCategory(raw_cat)
            except ValueError:
                raise ValueError(
                    f"Query '{qid}' has invalid category '{raw_cat}'. "
                    f"Allowed: {[c.value for c in QueryCategory]}."
                )

            raw_hn = q_data.get("hard_negative_chunk_ids", [])
            if not isinstance(raw_hn, (list, tuple)):
                raise ValueError(f"Query '{qid}' hard_negative_chunk_ids must be a list.")
            clean_hn = tuple(str(x) for x in raw_hn)

            notes = q_data.get("notes")

            parsed_queries.append(
                RetrievalGoldenQuery(
                    query_id=qid,
                    query=q_text,
                    relevance=clean_rel,
                    hard_negative_chunk_ids=clean_hn,
                    difficulty=diff_enum,
                    category=cat_enum,
                    notes=str(notes) if notes is not None else None,
                )
            )

        return RetrievalGoldenDataset(
            dataset_name=str(dataset_name).strip(),
            version=str(data.get("version", "v1")),
            description=str(data.get("description", "")),
            benchmark_type=str(data.get("benchmark_type", "PRODUCTION_GOLDEN_DATASET")),
            corpus_id=data.get("corpus_id"),
            queries=tuple(parsed_queries),
        )

    @staticmethod
    def validate_against_corpus(
        dataset: RetrievalGoldenDataset,
        corpus_chunk_ids: Sequence[str],
        strict: bool = True,
    ) -> list[str]:
        """Validates that all ground truth and hard negative chunk IDs exist in the active corpus."""
        existing_set = set(corpus_chunk_ids)
        missing_chunk_ids: list[str] = []

        for q in dataset.queries:
            for cid in q.relevance.keys():
                if cid not in existing_set:
                    missing_chunk_ids.append(cid)
            for cid in q.hard_negative_chunk_ids:
                if cid not in existing_set:
                    missing_chunk_ids.append(cid)

        unique_missing = sorted(set(missing_chunk_ids))
        if unique_missing and strict:
            raise ValueError(
                f"Dataset contains {len(unique_missing)} chunk IDs not present in corpus: {unique_missing[:5]}..."
            )
        return unique_missing


class RetrievalDatasetLoader:
    """Loads and validates retrieval benchmark datasets from filesystem or memory."""

    @classmethod
    def load_from_dict(cls, data: dict[str, Any]) -> RetrievalGoldenDataset:
        return RetrievalDatasetValidator.validate_dict(data)

    @classmethod
    def load_from_file(cls, path: str | Path) -> RetrievalGoldenDataset:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Benchmark dataset file not found at '{p}'.")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.load_from_dict(data)


# =============================================================================
# Benchmark Runner
# =============================================================================


class RetrievalBenchmarkRunner:
    """Executes Golden Retrieval benchmark suites against production retrieval services."""

    def __init__(
        self,
        session: Session,
        task_id: str,
        source_scope_ids: Sequence[str] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        retrieval_policy_version: str = "hybrid_rrf_v1",
        min_vector_similarity: float = 0.2,
    ) -> None:
        self.session = session
        self.task_id = task_id
        self.source_scope_ids = tuple(source_scope_ids) if source_scope_ids else ()
        self.embedding_provider = embedding_provider
        self.retrieval_policy_version = retrieval_policy_version
        self.min_vector_similarity = min_vector_similarity

        self.retrieval_service = KnowledgeRetrievalService(
            session=session,
            retrieval_policy_version=retrieval_policy_version,
            embedding_provider=embedding_provider,
            min_vector_similarity=min_vector_similarity,
        )

    def run_benchmark(
        self,
        dataset: RetrievalGoldenDataset,
        modes: Sequence[str] = ("BM25_ONLY", "DENSE_ONLY", "HYBRID"),
        benchmark_name: str = "retrieval_benchmark_run",
    ) -> RetrievalBenchmarkReport:
        """Executes benchmark over the specified modes with a single top_k=5 retrieval call per query per mode."""
        eval_modes = tuple(modes)
        mode_latencies: dict[str, list[float]] = {m: [] for m in eval_modes}
        mode_metrics: dict[str, dict[str, list[float]]] = {
            m: {
                "recall@1": [],
                "recall@3": [],
                "recall@5": [],
                "precision@5": [],
                "mrr": [],
                "ndcg@5": [],
            }
            for m in eval_modes
        }

        # Track hits per query across modes for cross-mode failure diagnostics
        # query_id -> mode -> bool (hit in top 5)
        query_mode_hits: dict[str, dict[str, bool]] = {q.query_id: {} for q in dataset.queries}
        raw_query_runs: dict[str, dict[str, dict[str, Any]]] = {q.query_id: {} for q in dataset.queries}

        # 1. Execute retrieval calls per query per mode
        for q in dataset.queries:
            rel_chunks = q.relevant_chunk_ids
            for mode in eval_modes:
                t0 = time.perf_counter()
                snapshot = self.retrieval_service.retrieve(
                    task_id=self.task_id,
                    query=q.query,
                    top_k=5,
                    source_scope_ids=self.source_scope_ids if self.source_scope_ids else None,
                    auto_create_evidence_items=False,
                    retrieval_mode=mode,
                )
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0

                ranked_cids = [c.chunk_id for c in snapshot.candidates]
                hit_in_top5 = bool(set(ranked_cids[:5]) & rel_chunks) if rel_chunks else False
                query_mode_hits[q.query_id][mode] = hit_in_top5

                raw_query_runs[q.query_id][mode] = {
                    "latency_ms": latency_ms,
                    "snapshot": snapshot,
                    "ranked_cids": ranked_cids,
                }

        # 2. Compute metrics, raw items, and identify failure cases
        all_per_query_results: list[PerQueryBenchmarkResult] = []
        failure_candidates: list[dict[str, Any]] = []

        calc = RetrievalMetricsCalculator
        for q in dataset.queries:
            rel_map = q.relevance
            bm25_hit = query_mode_hits[q.query_id].get("BM25_ONLY")
            dense_hit = query_mode_hits[q.query_id].get("DENSE_ONLY")

            for mode in eval_modes:
                run_data = raw_query_runs[q.query_id][mode]
                lat_ms = run_data["latency_ms"]
                snapshot = run_data["snapshot"]
                ranked_cids = run_data["ranked_cids"]

                r1 = calc.compute_recall_at_k(ranked_cids, rel_map, k=1)
                r3 = calc.compute_recall_at_k(ranked_cids, rel_map, k=3)
                r5 = calc.compute_recall_at_k(ranked_cids, rel_map, k=5)
                p5 = calc.compute_precision_at_k(ranked_cids, rel_map, k=5)
                mrr = calc.compute_mrr(ranked_cids, rel_map)
                ndcg5 = calc.compute_ndcg_at_k(ranked_cids, rel_map, k=5)

                mode_latencies[mode].append(lat_ms)
                mode_metrics[mode]["recall@1"].append(r1)
                mode_metrics[mode]["recall@3"].append(r3)
                mode_metrics[mode]["recall@5"].append(r5)
                mode_metrics[mode]["precision@5"].append(p5)
                mode_metrics[mode]["mrr"].append(mrr)
                mode_metrics[mode]["ndcg@5"].append(ndcg5)

                failures = calc.identify_failure_types(
                    ranked_chunk_ids=ranked_cids,
                    relevance_map=rel_map,
                    hard_negatives=q.hard_negative_chunk_ids,
                    mode=mode,
                    bm25_hit_top5=bm25_hit,
                    dense_hit_top5=dense_hit,
                )

                # Build PerQueryRankedItem records
                ranked_items: list[PerQueryRankedItem] = []
                hn_set = set(q.hard_negative_chunk_ids)
                for cand in snapshot.candidates:
                    grade = rel_map.get(cand.chunk_id, 0)
                    ranked_items.append(
                        PerQueryRankedItem(
                            rank=cand.rank,
                            chunk_id=cand.chunk_id,
                            score=cand.score,
                            retrieval_method=cand.retrieval_method,
                            expected_relevance_grade=grade,
                            is_hit=grade >= RELEVANT_GRADE_THRESHOLD,
                            is_hard_negative=cand.chunk_id in hn_set,
                        )
                    )

                metrics_dict = {
                    "recall@1": round(r1, 4),
                    "recall@3": round(r3, 4),
                    "recall@5": round(r5, 4),
                    "precision@5": round(p5, 4),
                    "mrr": round(mrr, 4),
                    "ndcg@5": round(ndcg5, 4),
                }

                res = PerQueryBenchmarkResult(
                    query_id=q.query_id,
                    query=q.query,
                    mode=mode,
                    category=q.category.value,
                    difficulty=q.difficulty.value,
                    latency_ms=round(lat_ms, 2),
                    ranked_items=tuple(ranked_items),
                    metrics=metrics_dict,
                    failure_types=tuple(failures),
                )
                all_per_query_results.append(res)

                if failures:
                    failure_candidates.append({
                        "query_id": q.query_id,
                        "query": q.query,
                        "mode": mode,
                        "category": q.category.value,
                        "failure_types": failures,
                        "returned_chunk_ids": ranked_cids,
                        "expected_relevant_chunk_ids": list(q.relevant_chunk_ids),
                    })

        # 3. Mode Aggregate Summaries
        summaries: dict[str, ModeBenchmarkSummary] = {}
        for mode in eval_modes:
            q_cnt = len(dataset.queries)
            m_dict = mode_metrics[mode]
            mean_lat, p95_lat = calc.compute_latency_percentiles(mode_latencies[mode])

            summaries[mode] = ModeBenchmarkSummary(
                mode=mode,
                query_count=q_cnt,
                recall_at_1=round(float(np.mean(m_dict["recall@1"])), 4) if q_cnt else 0.0,
                recall_at_3=round(float(np.mean(m_dict["recall@3"])), 4) if q_cnt else 0.0,
                recall_at_5=round(float(np.mean(m_dict["recall@5"])), 4) if q_cnt else 0.0,
                precision_at_5=round(float(np.mean(m_dict["precision@5"])), 4) if q_cnt else 0.0,
                mrr=round(float(np.mean(m_dict["mrr"])), 4) if q_cnt else 0.0,
                ndcg_at_5=round(float(np.mean(m_dict["ndcg@5"])), 4) if q_cnt else 0.0,
                mean_latency_ms=mean_lat,
                p95_latency_ms=p95_lat,
            )

        # 4. Safe Metadata (Zero Secrets)
        meta: dict[str, Any] = {
            "retrieval_policy_version": self.retrieval_policy_version,
            "min_vector_similarity": self.min_vector_similarity,
            "top_k": 5,
            "corpus_scope_count": len(self.source_scope_ids),
            "benchmark_type": dataset.benchmark_type,
        }
        if self.embedding_provider:
            meta["embedding_provider"] = self.embedding_provider.provider_name
            meta["embedding_model"] = self.embedding_provider.model_name
            meta["embedding_dimension"] = self.embedding_provider.dimension
        else:
            meta["embedding_provider"] = None

        return RetrievalBenchmarkReport(
            benchmark_name=benchmark_name,
            dataset_name=dataset.dataset_name,
            dataset_version=dataset.version,
            modes=summaries,
            per_query_results=tuple(all_per_query_results),
            failure_candidates=tuple(failure_candidates),
            metadata=meta,
        )
