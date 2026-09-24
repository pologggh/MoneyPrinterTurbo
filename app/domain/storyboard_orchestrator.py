from collections import defaultdict
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.content_plan import ContentPlanRevision
from app.domain.enums import StoryboardSnapshotState
from app.domain.plan_diff import (
    BeatChange,
    BeatChangeType,
    ShotReuseDecision,
    decide_shot_reuse,
    diff_content_plans,
)
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_agent import (
    StoryboardAgent,
    StoryboardExecutionResult,
    StoryboardPlanningInput,
)

# =============================================================================
# Exceptions
# =============================================================================


class StoryboardOrchestrationError(Exception):
    """Base exception for Storyboard Orchestration errors."""

    def __init__(self, message: str, code: str = "STORYBOARD_ORCHESTRATION_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


class StoryboardBaselineMismatchError(StoryboardOrchestrationError):
    """Raised when baseline snapshot and plan revisions do not match or are incompatible."""

    def __init__(self, message: str):
        super().__init__(message, code="STORYBOARD_BASELINE_MISMATCH")


class PlanNotFoundError(StoryboardOrchestrationError):
    """Raised when a specified ContentPlanRevision is not found in the repository."""

    def __init__(self, message: str):
        super().__init__(message, code="PLAN_NOT_FOUND")


class StoryboardSnapshotNotFoundError(StoryboardOrchestrationError):
    """Raised when a specified StoryboardSnapshot is not found in the repository."""

    def __init__(self, message: str):
        super().__init__(message, code="STORYBOARD_SNAPSHOT_NOT_FOUND")


class StoryboardBeatGenerationError(StoryboardOrchestrationError):
    """Raised when StoryboardAgent fails to generate shots for a required beat."""

    def __init__(self, message: str, beat_lineage_id: str | None = None):
        super().__init__(message, code="STORYBOARD_BEAT_GENERATION_FAILED")
        self.beat_lineage_id = beat_lineage_id


# =============================================================================
# Integration Contracts
# =============================================================================


class BeatStoryboardAction(str, Enum):
    REUSED = "REUSED"
    GENERATED = "GENERATED"
    REMOVED = "REMOVED"


class StoryboardBuildInput(BaseModel):
    """
    Input contract for the StoryboardPlanningOrchestrator.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    new_content_plan_revision_id: str = Field(
        description="ID of the newly planned ContentPlanRevision"
    )
    previous_content_plan_revision_id: str | None = Field(
        default=None,
        description="ID of previous ContentPlanRevision if revising",
    )
    previous_storyboard_snapshot_id: str | None = Field(
        default=None,
        description="ID of baseline StoryboardSnapshot corresponding to previous plan",
    )
    user_instruction: str | None = None
    target_aspect_ratio: str | None = None
    language: str | None = None
    source_grounded: bool = Field(
        default=True,
        description="Whether factual statements require verified evidence citations",
    )


class BeatStoryboardResult(BaseModel):
    """
    Detailed action and resulting shot revisions for a single beat lineage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    beat_lineage_id: str
    change_type: BeatChangeType
    action: BeatStoryboardAction
    shot_revision_ids: tuple[str, ...] = Field(default=())
    reason: str = ""


class StoryboardBuildStatistics(BaseModel):
    """
    Summary metrics of the storyboard build operation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_beats: int
    reused_beats: int
    regenerated_beats: int
    added_beats: int
    removed_beats: int
    reused_shots: int
    generated_shots: int


class StoryboardBuildResult(BaseModel):
    """
    Final output of the Storyboard Orchestrator containing the new Snapshot and metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    storyboard_snapshot: StoryboardSnapshot
    beat_results: tuple[BeatStoryboardResult, ...] = Field(default=())
    statistics: StoryboardBuildStatistics


# =============================================================================
# Storyboard Planning Orchestrator
# =============================================================================


class StoryboardOrchestrator:
    """
    Orchestrates the conversion of a ContentPlanRevision into a DRAFT StoryboardSnapshot,
    deciding which historical shots are reused and which beats require new generation.
    """

    def __init__(
        self,
        plan_repository: Any,
        shot_repository: Any,
        storyboard_repository: Any,
        storyboard_agent: StoryboardAgent | None = None,
    ):
        self._plan_repo = plan_repository
        self._shot_repo = shot_repository
        self._storyboard_repo = storyboard_repository
        self._agent = storyboard_agent or StoryboardAgent()

    def build_storyboard(
        self,
        input_data: StoryboardBuildInput,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> StoryboardBuildResult:
        start_ev = None
        if trace_writer is not None and trace_context is not None:
            from app.domain.trace import TraceEventType

            ctx = trace_context.with_ids(
                content_plan_revision_id=input_data.new_content_plan_revision_id
            )
            start_ev = trace_writer.start_event(
                context=ctx,
                event_type=TraceEventType.STORYBOARD_GENERATION_STARTED,
                attributes={"content_plan_revision_id": input_data.new_content_plan_revision_id},
            )

        # 1. Compatibility and Input Validation
        has_prev_plan = input_data.previous_content_plan_revision_id is not None
        has_prev_snap = input_data.previous_storyboard_snapshot_id is not None

        if has_prev_plan != has_prev_snap:
            raise StoryboardBaselineMismatchError(
                "Both previous_content_plan_revision_id and previous_storyboard_snapshot_id "
                "must be provided together, or neither."
            )

        new_plan = self._plan_repo.get_revision(input_data.new_content_plan_revision_id)
        if new_plan is None:
            raise PlanNotFoundError(
                f"New ContentPlanRevision '{input_data.new_content_plan_revision_id}' not found."
            )

        old_plan: ContentPlanRevision | None = None
        old_snapshot: StoryboardSnapshot | None = None

        if has_prev_plan and has_prev_snap:
            old_plan = self._plan_repo.get_revision(
                input_data.previous_content_plan_revision_id
            )
            if old_plan is None:
                raise PlanNotFoundError(
                    f"Previous ContentPlanRevision '{input_data.previous_content_plan_revision_id}' not found."
                )

            old_snapshot = self._storyboard_repo.get_snapshot(
                input_data.previous_storyboard_snapshot_id
            )
            if old_snapshot is None:
                raise StoryboardSnapshotNotFoundError(
                    f"Previous StoryboardSnapshot '{input_data.previous_storyboard_snapshot_id}' not found."
                )

            if (
                old_snapshot.content_plan_revision_id
                != old_plan.content_plan_revision_id
            ):
                raise StoryboardBaselineMismatchError(
                    f"Baseline mismatch: StoryboardSnapshot '{old_snapshot.storyboard_snapshot_id}' "
                    f"was created for plan '{old_snapshot.content_plan_revision_id}', but "
                    f"supplied previous plan was '{old_plan.content_plan_revision_id}'."
                )

        # 2. Compute Deterministic Diff
        if old_plan is not None:
            diff = diff_content_plans(old_plan, new_plan)
            changes_by_lineage = {c.beat_lineage_id: c for c in diff.beat_changes}
        else:
            changes_by_lineage = {
                b.beat_lineage_id: BeatChange(
                    beat_lineage_id=b.beat_lineage_id,
                    change_type=BeatChangeType.ADDED,
                    new_beat_instance_id=b.beat_id,
                    new_order=b.order,
                    reason="Initial plan creation",
                )
                for b in new_plan.beats
            }

        # 3. Load baseline snapshot shot revisions if revising
        baseline_revisions_by_lineage: dict[str, list[tuple[int, str]]] = defaultdict(
            list
        )
        if old_snapshot is not None:
            old_revisions = self._storyboard_repo.get_snapshot_shot_revisions(
                old_snapshot.storyboard_snapshot_id
            )
            for r in old_revisions:
                shot = self._shot_repo.get_shot(r.shot_id)
                local_order = shot.local_order if shot else 1
                baseline_revisions_by_lineage[r.beat_lineage_id].append(
                    (local_order, r.shot_revision_id)
                )

            # Sort each lineage's revisions by local_order
            for revs in baseline_revisions_by_lineage.values():
                revs.sort(key=lambda x: x[0])

        # 4. Orchestrate Beat Reuse / Generation (Outside DB Transaction)
        pending_shots_to_persist: list[Shot] = []
        pending_revisions_to_persist: list[ShotRevision] = []
        beat_results: list[BeatStoryboardResult] = []
        beat_ordered_shot_revision_ids: list[tuple[int, tuple[str, ...]]] = []

        reused_beats_count = 0
        regenerated_beats_count = 0
        added_beats_count = 0
        removed_beats_count = 0
        reused_shots_count = 0
        generated_shots_count = 0

        # Process each beat in the new plan
        for beat in sorted(new_plan.beats, key=lambda b: b.order):
            change = changes_by_lineage.get(beat.beat_lineage_id)
            change_type = (
                change.change_type if change is not None else BeatChangeType.ADDED
            )
            decision = decide_shot_reuse(change_type)

            if decision == ShotReuseDecision.REUSE_EXISTING_SHOTS:
                # UNCHANGED or MOVED: Reuse exact historical ShotRevisions
                reused_pairs = baseline_revisions_by_lineage.get(
                    beat.beat_lineage_id, []
                )
                reused_ids = tuple(rev_id for _, rev_id in reused_pairs)

                beat_results.append(
                    BeatStoryboardResult(
                        beat_lineage_id=beat.beat_lineage_id,
                        change_type=change_type,
                        action=BeatStoryboardAction.REUSED,
                        shot_revision_ids=reused_ids,
                        reason=(
                            "Reused historical shots"
                            if change_type == BeatChangeType.UNCHANGED
                            else f"Moved to order {beat.order}, reused historical shots"
                        ),
                    )
                )
                beat_ordered_shot_revision_ids.append((beat.order, reused_ids))
                reused_beats_count += 1
                reused_shots_count += len(reused_ids)

            else:
                # MODIFIED or ADDED: Generate completely new Shots via StoryboardAgent
                agent_input = StoryboardPlanningInput(
                    content_plan_revision_id=new_plan.content_plan_revision_id,
                    beat=beat,
                    video_title=new_plan.topic,
                    target_aspect_ratio=input_data.target_aspect_ratio,
                    user_instruction=input_data.user_instruction,
                    available_evidence_ids=beat.evidence_refs,
                    language=input_data.language,
                    source_grounded=input_data.source_grounded,
                )

                try:
                    agent_result: StoryboardExecutionResult = (
                        self._agent.generate_shots_for_beat(agent_input)
                    )
                except Exception as exc:
                    raise StoryboardBeatGenerationError(
                        f"Failed generating shots for beat '{beat.beat_id}' "
                        f"(lineage '{beat.beat_lineage_id}'): {exc}",
                        beat_lineage_id=beat.beat_lineage_id,
                    ) from exc

                new_rev_ids = tuple(
                    r.shot_revision_id for r in agent_result.shot_revisions
                )
                pending_shots_to_persist.extend(agent_result.shots)
                pending_revisions_to_persist.extend(agent_result.shot_revisions)

                action = BeatStoryboardAction.GENERATED
                reason = (
                    "Content modified, generated new shots"
                    if change_type == BeatChangeType.MODIFIED
                    else "New beat, generated new shots"
                )

                beat_results.append(
                    BeatStoryboardResult(
                        beat_lineage_id=beat.beat_lineage_id,
                        change_type=change_type,
                        action=action,
                        shot_revision_ids=new_rev_ids,
                        reason=reason,
                    )
                )
                beat_ordered_shot_revision_ids.append((beat.order, new_rev_ids))

                if change_type == BeatChangeType.MODIFIED:
                    regenerated_beats_count += 1
                else:
                    added_beats_count += 1
                generated_shots_count += len(new_rev_ids)

        # Record REMOVED beats from diff
        if old_plan is not None:
            for change in changes_by_lineage.values():
                if change.change_type == BeatChangeType.REMOVED:
                    beat_results.append(
                        BeatStoryboardResult(
                            beat_lineage_id=change.beat_lineage_id,
                            change_type=BeatChangeType.REMOVED,
                            action=BeatStoryboardAction.REMOVED,
                            shot_revision_ids=(),
                            reason="Beat removed in new revision",
                        )
                    )
                    removed_beats_count += 1

        # 5. Persist all newly generated shots/revisions (Atomically)
        for shot in pending_shots_to_persist:
            self._shot_repo.add_shot(shot)
        for rev in pending_revisions_to_persist:
            self._shot_repo.add_revision(rev)

        # 6. Assemble deterministic shot_revision_ids ordered by Beat.order then Shot.local_order
        final_shot_revision_ids: list[str] = []
        for _, shot_rev_ids in sorted(
            beat_ordered_shot_revision_ids, key=lambda x: x[0]
        ):
            final_shot_revision_ids.extend(shot_rev_ids)

        # 7. Create and persist immutable DRAFT StoryboardSnapshot
        new_snapshot = StoryboardSnapshot(
            storyboard_snapshot_id=str(uuid4()),
            content_plan_revision_id=new_plan.content_plan_revision_id,
            shot_revision_ids=tuple(final_shot_revision_ids),
            snapshot_state=StoryboardSnapshotState.DRAFT,
        )
        self._storyboard_repo.add_snapshot(new_snapshot)

        # 8. Compile Statistics and Return Result
        statistics = StoryboardBuildStatistics(
            total_beats=len(new_plan.beats),
            reused_beats=reused_beats_count,
            regenerated_beats=regenerated_beats_count,
            added_beats=added_beats_count,
            removed_beats=removed_beats_count,
            reused_shots=reused_shots_count,
            generated_shots=generated_shots_count,
        )

        if start_ev is not None and trace_writer is not None:
            from app.domain.trace import StoryboardGenerationTraceData, TraceEventStatus

            comp_data = StoryboardGenerationTraceData(
                content_plan_revision_id=new_plan.content_plan_revision_id,
                storyboard_snapshot_id=new_snapshot.storyboard_snapshot_id,
                shot_ids=tuple(s.shot_id for s in pending_shots_to_persist),
                shot_revision_ids=new_snapshot.shot_revision_ids,
                shot_count=len(new_snapshot.shot_revision_ids),
            )
            trace_writer.complete_event(
                event=start_ev,
                status=TraceEventStatus.SUCCEEDED,
                attributes_update=comp_data,
            )

        return StoryboardBuildResult(
            storyboard_snapshot=new_snapshot,
            beat_results=tuple(beat_results),
            statistics=statistics,
        )
