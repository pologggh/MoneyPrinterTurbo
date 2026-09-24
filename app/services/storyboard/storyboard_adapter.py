from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.content_plan import ContentPlanRevision
from app.domain.script import ScriptRevision
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_agent import (
    StoryboardAgent,
    StoryboardError,
    StoryboardExecutionResult,
    StoryboardPlanningInput,
)


class StoryboardAdapterError(Exception):
    """Base error for Storyboard adapter operations."""

    def __init__(self, message: str, code: str = "STORYBOARD_ADAPTER_ERROR") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class StoryboardScriptMappingError(StoryboardAdapterError):
    """Raised when script segments cannot be mapped to content beats or order is violated."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="STORYBOARD_SCRIPT_MAPPING_ERROR")


class StoryboardEvidenceScopeError(StoryboardAdapterError):
    """Raised when generated shots exceed or violate upstream script/beat evidence scope."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="STORYBOARD_EVIDENCE_SCOPE_ERROR")


class StoryboardDurationMismatchError(StoryboardAdapterError):
    """Raised when planned storyboard duration deviates from script target beyond tolerance."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="STORYBOARD_DURATION_MISMATCH")


class StoryboardAdapterResult(BaseModel):
    """Structured result of Storyboard planning from a ScriptRevision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot: StoryboardSnapshot
    shots: tuple[Shot, ...] = Field(default=())
    shot_revisions: tuple[ShotRevision, ...] = Field(default=())


class StoryboardAdapter:
    """
    Adapter connecting the structured, evidence-grounded Script stage output
    to the existing StoryboardAgent domain service without duplicating planning logic.
    """

    def __init__(
        self,
        storyboard_agent: StoryboardAgent | None = None,
        duration_tolerance_ratio: float = 0.25,
    ) -> None:
        self.storyboard_agent = storyboard_agent or StoryboardAgent()
        self.duration_tolerance_ratio = duration_tolerance_ratio

    def build_storyboard_from_script(
        self,
        plan: ContentPlanRevision,
        script: ScriptRevision,
        *,
        video_title: str | None = None,
        target_aspect_ratio: str | None = "16:9",
        user_instruction: str | None = None,
        language: str = "zh",
        shot_repository: Any | None = None,
        storyboard_repository: Any | None = None,
    ) -> StoryboardAdapterResult:
        if not script.segments:
            raise StoryboardScriptMappingError(
                f"ScriptRevision '{script.script_revision_id}' contains zero segments."
            )

        beats_by_id = {b.beat_id: b for b in plan.beats}
        beats_by_lineage = {b.beat_lineage_id: b for b in plan.beats}

        sorted_segments = sorted(script.segments, key=lambda s: s.order)

        all_shots: list[Shot] = []
        all_revisions: list[ShotRevision] = []

        for segment in sorted_segments:
            beat = beats_by_id.get(segment.content_beat_id)
            if beat is None and segment.beat_lineage_id:
                beat = beats_by_lineage.get(segment.beat_lineage_id)

            if beat is None:
                raise StoryboardScriptMappingError(
                    f"No ContentBeat found in plan '{plan.content_plan_revision_id}' for "
                    f"ScriptSegment '{segment.script_segment_id}' (content_beat_id='{segment.content_beat_id}')."
                )

            # Strict evidence scope: shot citations MUST be subset of segment evidence
            available_evidence = tuple(segment.evidence_refs)

            planning_input = StoryboardPlanningInput(
                content_plan_revision_id=plan.content_plan_revision_id,
                beat=beat,
                video_title=video_title or plan.topic,
                target_aspect_ratio=target_aspect_ratio,
                user_instruction=user_instruction,
                available_evidence_ids=available_evidence,
                language=language,
                source_grounded=True,
                authoritative_narration=segment.narration_text,
                script_segment_id=segment.script_segment_id,
            )

            try:
                agent_result: StoryboardExecutionResult = (
                    self.storyboard_agent.generate_shots_for_beat(
                        planning_input, shot_repository=shot_repository
                    )
                )
            except StoryboardError as exc:
                raise StoryboardAdapterError(
                    f"StoryboardAgent failed for beat '{beat.beat_id}': {exc}",
                    code=getattr(exc, "code", "STORYBOARD_AGENT_FAILED"),
                ) from exc

            if not agent_result.shots or not agent_result.shot_revisions:
                raise StoryboardAdapterError(
                    f"StoryboardAgent produced 0 shots for beat '{beat.beat_id}'."
                )

            # Validate each revision strictly against segment scope
            for rev in agent_result.shot_revisions:
                # 1. Evidence containment: rev.evidence_refs ⊆ segment.evidence_refs
                rev_ev_set = set(rev.evidence_refs)
                seg_ev_set = set(segment.evidence_refs)
                if not rev_ev_set.issubset(seg_ev_set):
                    excess = sorted(rev_ev_set - seg_ev_set)
                    raise StoryboardEvidenceScopeError(
                        f"Shot revision '{rev.shot_revision_id}' cites evidence {excess} "
                        f"not present in upstream ScriptSegment '{segment.script_segment_id}' scope {sorted(seg_ev_set)}."
                    )

                # 2. Narration non-empty
                if not rev.narration or not rev.narration.strip():
                    raise StoryboardAdapterError(
                        f"Shot revision '{rev.shot_revision_id}' has empty narration."
                    )

                # 3. Target duration positive
                if rev.target_duration <= 0:
                    raise StoryboardAdapterError(
                        f"Shot revision '{rev.shot_revision_id}' has invalid target_duration: {rev.target_duration}."
                    )

            all_shots.extend(agent_result.shots)
            all_revisions.extend(agent_result.shot_revisions)

        # Validate aggregate duration
        total_duration = sum(r.target_duration for r in all_revisions)
        target_duration = script.overall_target_duration
        max_deviation = max(3.0, target_duration * self.duration_tolerance_ratio)
        if abs(total_duration - target_duration) > max_deviation:
            raise StoryboardDurationMismatchError(
                f"Total planned shot duration ({total_duration:.1f}s) deviates from script target "
                f"({target_duration:.1f}s) beyond acceptable tolerance of {max_deviation:.1f}s."
            )

        # Create draft StoryboardSnapshot
        snapshot = self.storyboard_agent.create_draft_snapshot(
            content_plan_revision_id=plan.content_plan_revision_id,
            shot_revisions=all_revisions,
            storyboard_repository=storyboard_repository,
        )

        return StoryboardAdapterResult(
            snapshot=snapshot,
            shots=tuple(all_shots),
            shot_revisions=tuple(all_revisions),
        )
