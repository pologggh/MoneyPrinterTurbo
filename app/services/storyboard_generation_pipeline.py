from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.domain.content_plan import ContentPlanRevision
from app.domain.enums import StoryboardSnapshotState
from app.domain.planner import ContentPlanner, PlannerInput
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_agent import StoryboardAgent
from app.domain.storyboard_orchestrator import (
    StoryboardBuildInput,
    StoryboardBuildResult,
    StoryboardOrchestrator,
)
from app.persistence.repositories import (
    ContentPlanRepository,
    ShotRepository,
    StoryboardRepository,
)
from app.persistence.session import get_session


class EvidenceInputItem(BaseModel):
    """Structured factual evidence snippet provided for source-grounded generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Unique identifier for the evidence snippet",
    )
    content: str = Field(min_length=1, description="Factual evidence or context text")
    title: str | None = Field(default=None, description="Optional title or source name")
    url: str | None = Field(default=None, description="Optional source URL")


class StoryboardGenerationInput(BaseModel):
    """Input contract for generating a complete ContentPlan and Storyboard from a topic."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(description="User video topic or subject")
    target_video_duration: float = Field(
        default=60.0,
        gt=0,
        description="Target total video duration in seconds",
    )
    title: str | None = Field(default=None, description="Optional title override")
    user_instruction: str | None = Field(
        default=None,
        description="Custom creative guidelines or constraints",
    )
    target_aspect_ratio: str = Field(
        default="16:9",
        description="Aspect ratio, e.g. 16:9 or 9:16",
    )
    language: str = Field(default="zh", description="Video language code")
    source_grounded: bool = Field(
        default=False,
        description="If True, factual knowledge beats require explicit evidence references",
    )
    knowledge_context: tuple[str, ...] = Field(
        default=(),
        description="Optional source evidence texts or notes",
    )
    evidence_items: tuple[EvidenceInputItem, ...] = Field(
        default=(),
        description="Structured evidence items required when source_grounded=True",
    )

class StoryboardGenerationResult(BaseModel):
    """Output contract containing the created plan revision and draft storyboard snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content_plan_revision_id: str
    storyboard_snapshot_id: str
    topic: str
    target_video_duration: float
    target_aspect_ratio: str
    language: str
    beat_count: int
    shot_count: int
    shot_revision_ids: tuple[str, ...]
    state: StoryboardSnapshotState = StoryboardSnapshotState.DRAFT


class StoryboardGenerationPipeline:
    """
    Formal end-to-end pipeline:
    User Topic -> ContentPlanner -> ContentPlanRevision -> StoryboardOrchestrator -> StoryboardSnapshot (DRAFT)
    """

    def __init__(
        self,
        llm_caller: Callable[[str], str] | None = None,
        session_factory: Any | None = None,
    ):
        self._llm_caller = llm_caller
        self._session_factory = session_factory

    def generate(
        self,
        input_data: StoryboardGenerationInput,
        session: Session | None = None,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> StoryboardGenerationResult:
        """
        Executes the content planning and storyboard building sequence, persisting
        the results atomically in the PostgreSQL database.
        """
        if input_data.source_grounded and not input_data.evidence_items:
            raise ValueError(
                "source_grounded=True requires non-empty evidence_items; "
                "knowledge_context alone is not authoritative evidence"
            )
        evidence_ids = [item.evidence_id for item in input_data.evidence_items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_items must contain unique evidence_id values")

        if session is not None:
            return self._execute_in_session(
                session=session,
                input_data=input_data,
                trace_context=trace_context,
                trace_writer=trace_writer,
            )

        with get_session(self._session_factory) as db_session:
            result = self._execute_in_session(
                session=db_session,
                input_data=input_data,
                trace_context=trace_context,
                trace_writer=trace_writer,
            )
            db_session.commit()
            return result

    def _execute_in_session(
        self,
        session: Session,
        input_data: StoryboardGenerationInput,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> StoryboardGenerationResult:
        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        storyboard_repo = StoryboardRepository(session)

        # 1. Content Planning Phase
        evidence_ids = tuple(e.evidence_id for e in input_data.evidence_items)
        evidence_snippets = [
            f"[{e.evidence_id}] {e.content}" for e in input_data.evidence_items
        ]
        combined_knowledge = tuple(list(input_data.knowledge_context) + evidence_snippets)

        planner = ContentPlanner(
            repository=plan_repo,
            llm_caller=self._llm_caller,
        )
        planner_input = PlannerInput(
            topic=input_data.topic,
            target_video_duration=input_data.target_video_duration,
            title=input_data.title or input_data.topic,
            user_instruction=input_data.user_instruction,
            knowledge_context=combined_knowledge,
            available_evidence_ids=evidence_ids,
            source_grounded=input_data.source_grounded,
        )

        logger.info(
            f"Generating ContentPlan for topic='{input_data.topic}', "
            f"duration={input_data.target_video_duration}s"
        )
        plan_revision: ContentPlanRevision = planner.plan(
            input_data=planner_input,
            repository=plan_repo,
            trace_context=trace_context,
            trace_writer=trace_writer,
        )
        session.flush()

        # 2. Storyboard Orchestration Phase
        agent = StoryboardAgent(
            shot_repository=None,
            storyboard_repository=storyboard_repo,
            llm_caller=self._llm_caller,
        )
        orchestrator = StoryboardOrchestrator(
            plan_repository=plan_repo,
            shot_repository=shot_repo,
            storyboard_repository=storyboard_repo,
            storyboard_agent=agent,
        )
        build_input = StoryboardBuildInput(
            new_content_plan_revision_id=plan_revision.content_plan_revision_id,
            user_instruction=input_data.user_instruction,
            target_aspect_ratio=input_data.target_aspect_ratio,
            language=input_data.language,
            source_grounded=input_data.source_grounded,
        )

        logger.info(
            f"Building Storyboard for ContentPlanRevision '{plan_revision.content_plan_revision_id}'"
        )
        build_result: StoryboardBuildResult = orchestrator.build_storyboard(
            input_data=build_input,
            trace_context=trace_context,
            trace_writer=trace_writer,
        )
        session.flush()

        snapshot: StoryboardSnapshot = build_result.storyboard_snapshot

        return StoryboardGenerationResult(
            content_plan_revision_id=plan_revision.content_plan_revision_id,
            storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
            topic=plan_revision.topic,
            target_video_duration=plan_revision.overall_target_duration,
            target_aspect_ratio=input_data.target_aspect_ratio,
            language=input_data.language,
            beat_count=len(plan_revision.beats),
            shot_count=len(snapshot.shot_revision_ids),
            shot_revision_ids=snapshot.shot_revision_ids,
            state=snapshot.snapshot_state,
        )
