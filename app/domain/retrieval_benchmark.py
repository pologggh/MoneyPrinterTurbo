from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import math
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Domain Enums & Invariants
# =============================================================================


class QueryCategory(str, Enum):
    """Categorical classification of retrieval benchmark queries."""
    FACT = "FACT"
    CONCEPT = "CONCEPT"
    REASON = "REASON"
    COMPARISON = "COMPARISON"
    TERMINOLOGY = "TERMINOLOGY"
    PARAPHRASE = "PARAPHRASE"
    MULTI_EVIDENCE = "MULTI_EVIDENCE"
    HARD_NEGATIVE = "HARD_NEGATIVE"


class QueryDifficulty(str, Enum):
    """Subjective retrieval difficulty level."""
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


# Valid graded relevance scores:
# 3 = directly and substantially answers the query
# 2 = strongly relevant and partially answers
# 1 = useful supporting context
# 0 = irrelevant
VALID_RELEVANCE_GRADES = {0, 1, 2, 3}
RELEVANT_GRADE_THRESHOLD = 1


# =============================================================================
# Golden Dataset Models
# =============================================================================


class RetrievalGoldenQuery(BaseModel):
    """A benchmark query paired with graded relevance ground truth and negative examples."""
    model_config = ConfigDict(frozen=True)

    query_id: str
    query: str
    relevance: dict[str, int] = Field(default_factory=dict)
    hard_negative_chunk_ids: tuple[str, ...] = Field(default_factory=tuple)
    difficulty: QueryDifficulty = QueryDifficulty.MEDIUM
    category: QueryCategory = QueryCategory.FACT
    notes: str | None = None

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("query_id cannot be empty.")
        return s

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("query cannot be empty.")
        return s

    @field_validator("relevance")
    @classmethod
    def validate_relevance(cls, v: dict[str, int]) -> dict[str, int]:
        for cid, grade in v.items():
            if not isinstance(grade, int):
                raise ValueError(f"Relevance grade for chunk '{cid}' must be an integer, got {type(grade)}.")
            if grade not in VALID_RELEVANCE_GRADES:
                raise ValueError(
                    f"Invalid relevance grade {grade} for chunk '{cid}'. "
                    f"Must be one of {sorted(VALID_RELEVANCE_GRADES)}."
                )
        return v

    @property
    def relevant_chunk_ids(self) -> set[str]:
        """Returns set of chunk IDs that meet or exceed the relevant threshold (grade >= 1)."""
        return {cid for cid, grade in self.relevance.items() if grade >= RELEVANT_GRADE_THRESHOLD}


class RetrievalGoldenDataset(BaseModel):
    """A collection of graded benchmark queries against a specified corpus."""
    model_config = ConfigDict(frozen=True)

    dataset_name: str
    version: str = "v1"
    description: str = ""
    benchmark_type: str = "PRODUCTION_GOLDEN_DATASET"
    corpus_id: str | None = None
    queries: tuple[RetrievalGoldenQuery, ...] = Field(default_factory=tuple)

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("dataset_name cannot be empty.")
        return s

    @model_validator(mode="after")
    def validate_queries(self) -> RetrievalGoldenDataset:
        seen_ids: set[str] = set()
        for q in self.queries:
            if q.query_id in seen_ids:
                raise ValueError(f"Duplicate query_id '{q.query_id}' found in dataset '{self.dataset_name}'.")
            seen_ids.add(q.query_id)
        return self


# =============================================================================
# Benchmark Results & Failure Tracking Models
# =============================================================================


class PerQueryRankedItem(BaseModel):
    """Per-item ranked result record returned for a benchmark query."""
    model_config = ConfigDict(frozen=True)

    rank: int
    chunk_id: str
    score: float
    retrieval_method: str
    expected_relevance_grade: int = 0
    is_hit: bool = False
    is_hard_negative: bool = False


class PerQueryBenchmarkResult(BaseModel):
    """Full execution trace and evaluated metrics for a single query in a specific mode."""
    model_config = ConfigDict(frozen=True)

    query_id: str
    query: str
    mode: str
    category: str
    difficulty: str
    latency_ms: float
    ranked_items: tuple[PerQueryRankedItem, ...] = Field(default_factory=tuple)
    metrics: dict[str, float] = Field(default_factory=dict)
    failure_types: tuple[str, ...] = Field(default_factory=tuple)


class ModeBenchmarkSummary(BaseModel):
    """Aggregated evaluation metrics for a specific retrieval mode."""
    model_config = ConfigDict(frozen=True)

    mode: str
    query_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    precision_at_5: float
    mrr: float
    ndcg_at_5: float
    mean_latency_ms: float
    p95_latency_ms: float


class RetrievalBenchmarkReport(BaseModel):
    """Top-level benchmark report containing mode comparisons, per-query traces, and failure analysis."""
    model_config = ConfigDict(frozen=True)

    benchmark_name: str
    dataset_name: str
    dataset_version: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    modes: dict[str, ModeBenchmarkSummary] = Field(default_factory=dict)
    per_query_results: tuple[PerQueryBenchmarkResult, ...] = Field(default_factory=tuple)
    failure_candidates: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)
