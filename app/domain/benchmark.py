from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from app.domain.asset_router import ModelSelectionMode, RoutingStrategy
from app.domain.content_plan import ContentPlanRevision
from app.domain.enums import BeatType
from app.domain.storyboard import StoryboardSnapshot


class RealBenchmarkExternalCallsNotAuthorizedError(RuntimeError):
    """Raised when REAL benchmark mode is attempted without explicit allow_external_calls=True authorization."""


class BenchmarkExecutionMode(str, Enum):
    OFFLINE = "OFFLINE"
    REAL = "REAL"


class BenchmarkRunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_FAILURES = "COMPLETED_WITH_FAILURES"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class BenchmarkCaseStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    NEEDS_USER_ACTION = "NEEDS_USER_ACTION"
    CANCELLED = "CANCELLED"


class BenchmarkFailureStage(str, Enum):
    INPUT_VALIDATION = "INPUT_VALIDATION"
    PLANNING = "PLANNING"
    STORYBOARD = "STORYBOARD"
    STRUCTURAL_VALIDATION = "STRUCTURAL_VALIDATION"
    APPROVAL = "APPROVAL"
    ROUTING = "ROUTING"
    EXECUTION = "EXECUTION"
    EVALUATION = "EVALUATION"
    REMEDIATION = "REMEDIATION"
    INFRASTRUCTURE = "INFRASTRUCTURE"


@dataclass(frozen=True)
class KnowledgeFixtureRef:
    """Immutable reference to a stable local knowledge/evidence fixture."""
    fixture_id: str
    fixture_version: str
    file_path: str
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "fixture_version": self.fixture_version,
            "file_path": self.file_path,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class BenchmarkExpectedConstraints:
    """
    Deterministic structural expectations on the generated script/storyboard.
    Agents' exact output text can legitimately vary; these constraints enforce structure.
    """
    min_beats: int = 1
    max_beats: int = 10
    required_beat_types: tuple[BeatType, ...] = ()
    require_evidence_for_knowledge_beats: bool = True
    target_duration_tolerance: float = 2.0
    minimum_shots: int | None = None
    maximum_shots: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_beats": self.min_beats,
            "max_beats": self.max_beats,
            "required_beat_types": [
                bt.value if isinstance(bt, Enum) else str(bt)
                for bt in self.required_beat_types
            ],
            "require_evidence_for_knowledge_beats": self.require_evidence_for_knowledge_beats,
            "target_duration_tolerance": self.target_duration_tolerance,
            "minimum_shots": self.minimum_shots,
            "maximum_shots": self.maximum_shots,
        }


def compute_benchmark_case_fingerprint(
    case_key: str,
    case_version: str,
    topic: str,
    target_duration: float,
    user_instruction: str | None,
    knowledge_fixture_ref: KnowledgeFixtureRef,
    expected_constraints: BenchmarkExpectedConstraints,
    tags: tuple[str, ...],
) -> str:
    """
    Deterministic SHA-256 fingerprint for a BenchmarkCase based on canonical JSON.
    Excludes database IDs, timestamps, and generated outputs.
    """
    payload = {
        "case_key": case_key,
        "case_version": case_version,
        "topic": topic,
        "target_duration": round(float(target_duration), 4),
        "user_instruction": user_instruction or "",
        "knowledge_fixture": knowledge_fixture_ref.to_dict(),
        "expected_constraints": expected_constraints.to_dict(),
        "tags": sorted(tags),
    }
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


@dataclass(frozen=True)
class BenchmarkCase:
    """
    Versioned benchmark test case representing input + structural expectations.
    Does NOT contain generated outputs.
    """
    case_key: str
    case_version: str
    title: str
    topic: str
    target_duration: float
    knowledge_fixture_ref: KnowledgeFixtureRef
    expected_constraints: BenchmarkExpectedConstraints = field(default_factory=BenchmarkExpectedConstraints)
    user_instruction: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    benchmark_case_id: str = field(default_factory=lambda: f"bcase_{uuid4().hex[:16]}")
    content_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.content_fingerprint:
            fp = compute_benchmark_case_fingerprint(
                case_key=self.case_key,
                case_version=self.case_version,
                topic=self.topic,
                target_duration=self.target_duration,
                user_instruction=self.user_instruction,
                knowledge_fixture_ref=self.knowledge_fixture_ref,
                expected_constraints=self.expected_constraints,
                tags=self.tags,
            )
            object.__setattr__(self, "content_fingerprint", fp)


@dataclass(frozen=True)
class BenchmarkCaseRef:
    """Frozen reference to an exact case version and fingerprint inside a suite."""
    case_key: str
    case_version: str
    content_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_key": self.case_key,
            "case_version": self.case_version,
            "content_fingerprint": self.content_fingerprint,
        }


def compute_benchmark_suite_fingerprint(
    suite_key: str,
    suite_version: str,
    case_refs: tuple[BenchmarkCaseRef, ...],
) -> str:
    """
    Deterministic SHA-256 fingerprint for a BenchmarkSuite.
    Preserves explicit case ordering.
    """
    payload = {
        "suite_key": suite_key,
        "suite_version": suite_version,
        "cases": [ref.to_dict() for ref in case_refs],
    }
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


@dataclass(frozen=True)
class BenchmarkSuite:
    """
    Immutable/versioned benchmark suite freezing exact BenchmarkCase versions in fixed order.
    """
    suite_key: str
    suite_version: str
    benchmark_case_refs: tuple[BenchmarkCaseRef, ...]
    benchmark_suite_id: str = field(default_factory=lambda: f"bsuite_{uuid4().hex[:16]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    content_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.content_fingerprint:
            fp = compute_benchmark_suite_fingerprint(
                suite_key=self.suite_key,
                suite_version=self.suite_version,
                case_refs=self.benchmark_case_refs,
            )
            object.__setattr__(self, "content_fingerprint", fp)


@dataclass(frozen=True)
class BenchmarkVariant:
    """
    Describes the system configuration / strategy being benchmarked.
    """
    variant_key: str
    routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    model_selection_mode: ModelSelectionMode = ModelSelectionMode.AUTO
    configured_provider: str | None = None
    configured_model: str | None = None
    evaluation_policy_version: str = "v1.0"
    reuse_mode: str = "DEFAULT"
    application_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_key": self.variant_key,
            "routing_strategy": self.routing_strategy.value if hasattr(self.routing_strategy, "value") else str(self.routing_strategy),
            "model_selection_mode": self.model_selection_mode.value if hasattr(self.model_selection_mode, "value") else str(self.model_selection_mode),
            "configured_provider": self.configured_provider,
            "configured_model": self.configured_model,
            "evaluation_policy_version": self.evaluation_policy_version,
            "reuse_mode": self.reuse_mode,
            "application_metadata": self.application_metadata,
        }


@dataclass(frozen=True)
class BenchmarkProductionResultRefs:
    """Exact entity references captured from the executed production pipeline."""
    content_plan_revision_id: str | None = None
    approved_storyboard_snapshot_id: str | None = None
    asset_route_plan_id: str | None = None
    execution_run_id: str | None = None
    accepted_shot_asset_version_ids: tuple[str, ...] = ()
    final_evaluation_snapshot_ids: tuple[str, ...] = ()
    remediation_decision_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_plan_revision_id": self.content_plan_revision_id,
            "approved_storyboard_snapshot_id": self.approved_storyboard_snapshot_id,
            "asset_route_plan_id": self.asset_route_plan_id,
            "execution_run_id": self.execution_run_id,
            "accepted_shot_asset_version_ids": list(self.accepted_shot_asset_version_ids),
            "final_evaluation_snapshot_ids": list(self.final_evaluation_snapshot_ids),
            "remediation_decision_ids": list(self.remediation_decision_ids),
        }


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """
    Immutable record linking a BenchmarkCase run to its exact Phase 7.1 trace_id.
    """
    benchmark_run_id: str
    benchmark_case_id: str
    case_key: str
    case_version: str
    case_fingerprint: str
    trace_id: str
    status: BenchmarkCaseStatus
    started_at: datetime
    benchmark_case_result_id: str = field(default_factory=lambda: f"bcres_{uuid4().hex[:16]}")
    finished_at: datetime | None = None
    duration_ms: float | None = None
    failure_stage: BenchmarkFailureStage | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    production_result_refs: BenchmarkProductionResultRefs = field(default_factory=BenchmarkProductionResultRefs)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkRun:
    """
    Immutable record orchestrating the execution of a BenchmarkSuite under a BenchmarkVariant.
    """
    benchmark_suite_id: str
    suite_key: str
    suite_version: str
    suite_fingerprint: str
    variant: BenchmarkVariant
    benchmark_run_id: str = field(default_factory=lambda: f"brun_{uuid4().hex[:16]}")
    execution_mode: BenchmarkExecutionMode = BenchmarkExecutionMode.OFFLINE
    status: BenchmarkRunStatus = BenchmarkRunStatus.CREATED
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    application_version: str = "1.3.6"
    git_commit: str | None = None
    total_cases: int = 0
    completed_cases: int = 0
    failed_cases: int = 0


def evaluate_benchmark_structural_constraints(
    plan_rev: ContentPlanRevision,
    storyboard_snapshot: StoryboardSnapshot | None,
    constraints: BenchmarkExpectedConstraints,
    target_case_duration: float,
) -> tuple[bool, list[str]]:
    """
    Evaluates generated ContentPlanRevision and StoryboardSnapshot against BenchmarkExpectedConstraints.
    Deterministic, code-only inspection (zero LLM grading).
    """
    errors: list[str] = []

    # 1. Beat count
    beat_count = len(plan_rev.beats)
    if beat_count < constraints.min_beats:
        errors.append(f"Beat count ({beat_count}) below minimum ({constraints.min_beats})")
    if beat_count > constraints.max_beats:
        errors.append(f"Beat count ({beat_count}) exceeds maximum ({constraints.max_beats})")

    # 2. Required beat types
    actual_beat_types = {b.beat_type for b in plan_rev.beats}
    for req_type in constraints.required_beat_types:
        if req_type not in actual_beat_types:
            errors.append(f"Required BeatType '{req_type.value}' missing from content plan")

    # 3. KNOWLEDGE beats evidence check
    if constraints.require_evidence_for_knowledge_beats:
        for b in plan_rev.beats:
            if b.beat_type == BeatType.KNOWLEDGE and not b.evidence_refs:
                beat_desc = getattr(b, "intent", b.beat_id)
                errors.append(f"KNOWLEDGE beat '{beat_desc}' (id: {b.beat_id}) is missing evidence_refs")

    # 4. Duration tolerance
    plan_duration = sum(b.target_duration for b in plan_rev.beats)
    diff = abs(plan_duration - target_case_duration)
    if diff > constraints.target_duration_tolerance:
        errors.append(
            f"Plan total duration ({plan_duration:.1f}s) differs from target ({target_case_duration:.1f}s) "
            f"by {diff:.1f}s, exceeding tolerance ({constraints.target_duration_tolerance:.1f}s)"
        )

    # 5. Shot count bounds (if storyboard exists)
    if storyboard_snapshot is not None:
        shot_count = len(storyboard_snapshot.shot_revision_ids)
        if constraints.minimum_shots is not None and shot_count < constraints.minimum_shots:
            errors.append(f"Shot count ({shot_count}) below minimum ({constraints.minimum_shots})")
        if constraints.maximum_shots is not None and shot_count > constraints.maximum_shots:
            errors.append(f"Shot count ({shot_count}) exceeds maximum ({constraints.maximum_shots})")

    return (len(errors) == 0, errors)
