"""Domain models and services for Storyboard Human Approval and Beat-level Replan.

Phase 3.3 implementation adhering to:
- Immutable approval deriving NEW APPROVED StoryboardSnapshot.
- Dedicated StoryboardApprovalRecord linking draft and approved snapshots.
- Beat-level Replan producing completely NEW Shot identities for the replanned beat.
- Exact reuse of unchanged Beat revisions.
- Guardrails preventing ContentPlan mutation from storyboard replan.
"""

import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import StoryboardSnapshotState
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_agent import (
    StoryboardError,
    StoryboardPlanningInput,
)
from app.domain.storyboard_editing import (
    StoryboardEditingError,
    StoryboardNotFoundError,
)

# =============================================================================
# Exceptions
# =============================================================================


class StoryboardApprovalError(StoryboardEditingError):
    """Base exception for storyboard approval errors."""

    def __init__(self, message: str, code: str = "STORYBOARD_APPROVAL_ERROR"):
        super().__init__(message, code=code)


class InvalidSnapshotStateForApprovalError(StoryboardApprovalError):
    """Raised when attempting to approve a snapshot that is not in DRAFT state."""

    def __init__(self, message: str):
        super().__init__(message, code="INVALID_SNAPSHOT_STATE_FOR_APPROVAL")


class ApprovalValidationError(StoryboardApprovalError):
    """Raised when a storyboard snapshot fails structural or semantic validation for approval."""

    def __init__(self, message: str):
        super().__init__(message, code="APPROVAL_VALIDATION_ERROR")


class StaleApprovalSnapshotError(StoryboardApprovalError):
    """Raised when approval is attempted on a stale snapshot ID."""

    def __init__(self, message: str):
        super().__init__(message, code="STALE_APPROVAL_SNAPSHOT")


class StoryboardBeatReplanError(StoryboardApprovalError):
    """Raised when beat-level replanning fails."""

    def __init__(self, message: str, code: str = "STORYBOARD_REPLAN_FAILED"):
        super().__init__(message, code=code)


class ContentReplanRequiredError(StoryboardBeatReplanError):
    """Raised when user replan request requires changing the ContentPlan rather than shots."""

    def __init__(self, message: str):
        super().__init__(message, code="CONTENT_REPLAN_REQUIRED")


# =============================================================================
# Domain Models & Contracts
# =============================================================================


class StoryboardApprovalRecord(BaseModel):
    """
    Immutable audit record representing a human confirmation of a DRAFT Storyboard.
    Links the source draft snapshot to the resulting frozen approved snapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    storyboard_approval_id: str = Field(default_factory=lambda: str(uuid4()))
    source_draft_snapshot_id: str = Field(
        description="ID of the working DRAFT StoryboardSnapshot that was reviewed"
    )
    approved_storyboard_snapshot_id: str = Field(
        description="ID of the new APPROVED StoryboardSnapshot created by this action"
    )
    content_plan_revision_id: str = Field(
        description="ID of the ContentPlanRevision associated with this approval"
    )
    exact_shot_revision_ids: tuple[str, ...] = Field(
        description="Exact frozen sequence of ShotRevision IDs included in the approved snapshot"
    )
    approved_by: str = Field(
        default="local_user",
        description="User identity or identifier that performed the approval",
    )
    approved_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the approval was committed",
    )
    user_note: str | None = Field(
        default=None,
        description="Optional concise human note or sign-off comment",
    )


class StoryboardBeatReplanInput(BaseModel):
    """
    Input contract for requesting a controlled agent replan of a single ContentBeat.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_storyboard_snapshot_id: str = Field(
        description="ID of the current StoryboardSnapshot being edited"
    )
    beat_lineage_id: str = Field(
        description="Lineage ID of the single ContentBeat to replan"
    )
    user_instruction: str | None = Field(
        default=None,
        description="Optional guidance for the StoryboardAgent (e.g. '用图解表达')",
    )
    video_title: str | None = None
    target_aspect_ratio: str | None = None
    language: str | None = None
    source_grounded: bool = True

    @model_validator(mode="before")
    @classmethod
    def _resolve_aliases(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and "target_beat_lineage_id" in data
            and "beat_lineage_id" not in data
        ):
            data = dict(data)
            data["beat_lineage_id"] = data.pop("target_beat_lineage_id")
        return data

    @property
    def target_beat_lineage_id(self) -> str:
        return self.beat_lineage_id



class StoryboardBeatReplanResult(BaseModel):
    """
    Structured domain result after executing a beat-level agent replan.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    previous_storyboard_snapshot_id: str
    new_storyboard_snapshot_id: str
    beat_lineage_id: str
    old_shot_revision_ids: tuple[str, ...]
    new_shot_revision_ids: tuple[str, ...]
    generated_shot_count: int
    state: StoryboardSnapshotState = StoryboardSnapshotState.DRAFT


# =============================================================================
# Helper Check for Content Plan Scope Violations
# =============================================================================

# Keywords indicating user instruction requires changing the ContentPlan itself
_CONTENT_PLAN_MUTATION_PATTERNS = [
    r"(新增|添加|增加).*(段落|章节|知识点|大纲|beat|内容)",
    r"(删除|移除|去掉).*(段落|章节|知识点|大纲|beat|段)",
    r"(修改|换个|更换|重写|重新规划|重构).*(主题|大纲|结构|整体)",
    r".*(主题|大纲|结构|整体).*(换成|改成|重写|修改|改变|换)",
    r"调整.*(段落|章节|beat).*顺序",
    r"改变.*(段落|章节|beat).*意图",
    r"add\s+.*(beat|topic|section|knowledge)",
    r"(remove|delete)\s+.*(beat|topic|section|knowledge)",
    r"reorder\s+.*(beat|topic|section)",
    r"change\s+.*(topic|plan|intent)",
]


def _check_content_replan_required(instruction: str | None) -> None:
    if not instruction:
        return
    text = instruction.strip().lower()
    for pattern in _CONTENT_PLAN_MUTATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise ContentReplanRequiredError(
                "CONTENT_REPLAN_REQUIRED: The requested instruction requires modifying "
                "the ContentPlan (structure/intent/beats). Please use Content Planner to replan the topic."
            )


# =============================================================================
# Storyboard Approval Outcome
# =============================================================================


class StoryboardApprovalOutcome(tuple):
    """
    Tuple of (StoryboardSnapshot, StoryboardApprovalRecord) supporting both
    unpacking as (snapshot, record) and direct attribute access to record properties.
    """

    def __new__(
        cls,
        approved_snapshot: StoryboardSnapshot,
        approval_record: StoryboardApprovalRecord,
    ):
        return super().__new__(cls, (approved_snapshot, approval_record))

    @property
    def approved_snapshot(self) -> StoryboardSnapshot:
        return self[0]

    @property
    def approval_record(self) -> StoryboardApprovalRecord:
        return self[1]

    @property
    def storyboard_approval_id(self) -> str:
        return self[1].storyboard_approval_id

    @property
    def source_draft_snapshot_id(self) -> str:
        return self[1].source_draft_snapshot_id

    @property
    def approved_storyboard_snapshot_id(self) -> str:
        return self[1].approved_storyboard_snapshot_id

    @property
    def content_plan_revision_id(self) -> str:
        return self[1].content_plan_revision_id

    @property
    def exact_shot_revision_ids(self) -> tuple[str, ...]:
        return self[1].exact_shot_revision_ids

    @property
    def approved_by(self) -> str:
        return self[1].approved_by

    @property
    def approved_at(self) -> datetime:
        return self[1].approved_at

    @property
    def user_note(self) -> str | None:
        return self[1].user_note


# =============================================================================
# Storyboard Approval Service
# =============================================================================


class StoryboardApprovalService:
    """
    Application service orchestrating Storyboard Human Approval and Beat-level Agent Replan.
    """

    def __init__(
        self,
        plan_repository: Any,
        shot_repository: Any,
        storyboard_repository: Any,
        approval_repository: Any,
        agent: Any | None = None,
    ):
        self._plan_repo = plan_repository
        self._shot_repo = shot_repository
        self._storyboard_repo = storyboard_repository
        self._approval_repo = approval_repository
        self._agent = agent

    def approve_storyboard(
        self,
        storyboard_snapshot_id: str | None = None,
        snapshot_id: str | None = None,
        approved_by: str = "local_user",
        user_note: str | None = None,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> StoryboardApprovalOutcome:
        """
        Validates and approves a DRAFT StoryboardSnapshot.

        Creates a NEW immutable APPROVED StoryboardSnapshot that freezes the exact
        ShotRevision IDs of the source draft, and persists a StoryboardApprovalRecord.
        The historical DRAFT snapshot is preserved untouched.
        """
        target_id = storyboard_snapshot_id or snapshot_id
        if not target_id:
            raise StoryboardNotFoundError("No storyboard snapshot ID provided.")

        source_snapshot: StoryboardSnapshot | None = (
            self._storyboard_repo.get_snapshot(target_id)
        )
        if source_snapshot is None:
            raise StoryboardNotFoundError(
                f"StoryboardSnapshot '{target_id}' not found."
            )

        if source_snapshot.snapshot_state != StoryboardSnapshotState.DRAFT:
            raise InvalidSnapshotStateForApprovalError(
                f"Cannot approve snapshot '{target_id}' because its state is "
                f"'{source_snapshot.snapshot_state.value}', expected DRAFT."
            )

        if not source_snapshot.shot_revision_ids:
            raise ApprovalValidationError(
                f"StoryboardSnapshot '{target_id}' contains zero shots and cannot be approved."
            )

        plan = self._plan_repo.get_revision(source_snapshot.content_plan_revision_id)
        if plan is None:
            raise ApprovalValidationError(
                f"Referenced ContentPlanRevision '{source_snapshot.content_plan_revision_id}' not found."
            )

        # Retrieve exact frozen ShotRevisions referenced in this snapshot
        frozen_revisions = self._storyboard_repo.get_snapshot_shot_revisions(
            source_snapshot.storyboard_snapshot_id
        )
        if len(frozen_revisions) != len(source_snapshot.shot_revision_ids):
            raise ApprovalValidationError(
                "Mismatch between snapshot revision count and resolved shot revisions."
            )

        # Validate non-empty narration on all shots
        for rev in frozen_revisions:
            if not rev.narration or not rev.narration.strip():
                raise ApprovalValidationError(
                    f"ShotRevision '{rev.shot_revision_id}' has empty narration. "
                    "All shots must have valid narration before approval."
                )

        # Validate shot ordering & beat coverage
        shots_by_lineage: dict[str, list[Shot]] = {}
        for rev in frozen_revisions:
            shot: Shot | None = self._shot_repo.get_shot(rev.shot_id)
            if shot is None:
                raise ApprovalValidationError(
                    f"Shot entity '{rev.shot_id}' not found."
                )
            shots_by_lineage.setdefault(rev.beat_lineage_id, []).append(shot)

        # Each beat in the content plan must have at least one shot
        for beat in plan.beats:
            beat_shots = shots_by_lineage.get(beat.beat_lineage_id, [])
            if not beat_shots:
                raise ApprovalValidationError(
                    f"ContentBeat '{beat.beat_lineage_id}' (order {beat.order}) "
                    "has no shots in this storyboard."
                )
            # Local orders within each beat must be 1..N contiguous
            local_orders = sorted(s.local_order for s in beat_shots)
            expected_orders = list(range(1, len(beat_shots) + 1))
            if local_orders != expected_orders:
                raise ApprovalValidationError(
                    f"Beat '{beat.beat_lineage_id}' has non-contiguous or invalid shot orders: "
                    f"{local_orders}, expected {expected_orders}."
                )

        # Create NEW immutable APPROVED snapshot with exact same shot_revision_ids
        approved_snapshot = StoryboardSnapshot(
            content_plan_revision_id=source_snapshot.content_plan_revision_id,
            snapshot_state=StoryboardSnapshotState.APPROVED,
            shot_revision_ids=source_snapshot.shot_revision_ids,
        )
        self._storyboard_repo.add_snapshot(approved_snapshot)

        # Create and persist StoryboardApprovalRecord
        approval_record = StoryboardApprovalRecord(
            source_draft_snapshot_id=source_snapshot.storyboard_snapshot_id,
            approved_storyboard_snapshot_id=approved_snapshot.storyboard_snapshot_id,
            content_plan_revision_id=source_snapshot.content_plan_revision_id,
            exact_shot_revision_ids=source_snapshot.shot_revision_ids,
            approved_by=approved_by,
            user_note=user_note,
        )
        self._approval_repo.add_approval_record(approval_record)

        if trace_writer is not None and trace_context is not None:
            from app.domain.trace import (
                StoryboardApprovalTraceData,
                TraceEventStatus,
                TraceEventType,
            )

            ctx = trace_context.with_ids(
                content_plan_revision_id=source_snapshot.content_plan_revision_id,
                storyboard_snapshot_id=approved_snapshot.storyboard_snapshot_id,
            )
            appr_data = StoryboardApprovalTraceData(
                draft_snapshot_id=source_snapshot.storyboard_snapshot_id,
                approved_snapshot_id=approved_snapshot.storyboard_snapshot_id,
                approval_record_id=approval_record.storyboard_approval_id,
                shot_count=len(approved_snapshot.shot_revision_ids),
                approval_timestamp=approval_record.approved_at,
            )
            trace_writer.record_event(
                context=ctx,
                event_type=TraceEventType.STORYBOARD_APPROVED,
                status=TraceEventStatus.SUCCEEDED,
                attributes=appr_data,
            )

        return StoryboardApprovalOutcome(approved_snapshot, approval_record)

    def replan_beat(
        self,
        input_data: StoryboardBeatReplanInput,
        storyboard_agent: Any | None = None,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> StoryboardBeatReplanResult:
        """
        Executes a controlled Agent replan for ONE ContentBeat.

        Generates completely NEW Shot entities for the target beat.
        Reuses exact frozen ShotRevision references for all other beats.
        Produces a NEW DRAFT StoryboardSnapshot, preserving base snapshot in history.
        """
        agent = storyboard_agent or self._agent
        if agent is None:
            from app.domain.storyboard_agent import StoryboardAgent

            agent = StoryboardAgent(shot_repository=self._shot_repo)

        base_snapshot: StoryboardSnapshot | None = (
            self._storyboard_repo.get_snapshot(input_data.base_storyboard_snapshot_id)
        )
        if base_snapshot is None:
            raise StoryboardNotFoundError(
                f"StoryboardSnapshot '{input_data.base_storyboard_snapshot_id}' not found."
            )

        plan = self._plan_repo.get_revision(base_snapshot.content_plan_revision_id)
        if plan is None:
            raise StoryboardNotFoundError(
                f"ContentPlanRevision '{base_snapshot.content_plan_revision_id}' not found."
            )

        target_beat = next(
            (b for b in plan.beats if b.beat_lineage_id == input_data.beat_lineage_id),
            None,
        )
        if target_beat is None:
            raise StoryboardBeatReplanError(
                f"Beat lineage '{input_data.beat_lineage_id}' not found in ContentPlan."
            )

        # Check for semantic content change requests
        _check_content_replan_required(input_data.user_instruction)

        # Load all frozen revisions of base snapshot
        base_revisions = self._storyboard_repo.get_snapshot_shot_revisions(
            base_snapshot.storyboard_snapshot_id
        )

        old_shot_revisions = [
            rev
            for rev in base_revisions
            if rev.beat_lineage_id == input_data.beat_lineage_id
        ]
        old_shot_revision_ids = tuple(
            rev.shot_revision_id for rev in old_shot_revisions
        )

        # Prepare planning input for StoryboardAgent
        planning_input = StoryboardPlanningInput(
            content_plan_revision_id=plan.content_plan_revision_id,
            beat=target_beat,
            video_title=input_data.video_title or plan.topic,
            target_aspect_ratio=input_data.target_aspect_ratio,
            user_instruction=input_data.user_instruction,
            language=input_data.language,
            source_grounded=input_data.source_grounded,
        )

        try:
            if hasattr(agent, "plan_beat"):
                execution_result = agent.plan_beat(planning_input)
            else:
                execution_result = agent.generate_shots_for_beat(planning_input)
        except StoryboardError as exc:
            raise StoryboardBeatReplanError(
                f"STORYBOARD_REPLAN_FAILED: StoryboardAgent failed for beat "
                f"'{input_data.beat_lineage_id}': {exc.message}"
            ) from exc
        except Exception as exc:
            raise StoryboardBeatReplanError(
                f"STORYBOARD_REPLAN_FAILED: Unexpected error during beat replan: {exc}"
            ) from exc

        # Persist new Shot entities and initial ShotRevisions
        new_shots: list[Shot] = []
        for shot in execution_result.shots:
            self._shot_repo.add_shot(shot)
            new_shots.append(shot)

        new_revisions: list[ShotRevision] = []
        for rev in execution_result.shot_revisions:
            self._shot_repo.add_revision(rev)
            new_revisions.append(rev)

        new_shot_revision_ids = tuple(rev.shot_revision_id for rev in new_revisions)

        # Reassemble global sequence of shot_revision_ids by Beat.order
        # Other beats reuse exact frozen revisions from base_snapshot
        final_revision_ids: list[str] = []
        for beat in sorted(plan.beats, key=lambda b: b.order):
            if beat.beat_lineage_id == input_data.beat_lineage_id:
                final_revision_ids.extend(new_shot_revision_ids)
            else:
                other_beat_rev_ids = [
                    rev.shot_revision_id
                    for rev in base_revisions
                    if rev.beat_lineage_id == beat.beat_lineage_id
                ]
                final_revision_ids.extend(other_beat_rev_ids)

        # Persist new DRAFT snapshot
        new_snapshot = StoryboardSnapshot(
            content_plan_revision_id=plan.content_plan_revision_id,
            snapshot_state=StoryboardSnapshotState.DRAFT,
            shot_revision_ids=tuple(final_revision_ids),
        )
        self._storyboard_repo.add_snapshot(new_snapshot)

        if trace_writer is not None and trace_context is not None:
            from app.domain.trace import (
                StoryboardGenerationTraceData,
                TraceEventStatus,
                TraceEventType,
            )

            ctx = trace_context.with_ids(
                content_plan_revision_id=plan.content_plan_revision_id,
                storyboard_snapshot_id=new_snapshot.storyboard_snapshot_id,
            )
            replan_data = StoryboardGenerationTraceData(
                content_plan_revision_id=plan.content_plan_revision_id,
                storyboard_snapshot_id=new_snapshot.storyboard_snapshot_id,
                beat_lineage_id=input_data.beat_lineage_id,
                shot_ids=tuple(s.shot_id for s in new_shots),
                shot_revision_ids=new_snapshot.shot_revision_ids,
                shot_count=len(new_snapshot.shot_revision_ids),
            )
            trace_writer.record_event(
                context=ctx,
                event_type=TraceEventType.STORYBOARD_REPLAN_COMPLETED,
                status=TraceEventStatus.SUCCEEDED,
                attributes=replan_data,
            )

        return StoryboardBeatReplanResult(
            previous_storyboard_snapshot_id=base_snapshot.storyboard_snapshot_id,
            new_storyboard_snapshot_id=new_snapshot.storyboard_snapshot_id,
            beat_lineage_id=input_data.beat_lineage_id,
            old_shot_revision_ids=old_shot_revision_ids,
            new_shot_revision_ids=new_shot_revision_ids,
            generated_shot_count=len(new_revisions),
            state=StoryboardSnapshotState.DRAFT,
        )
