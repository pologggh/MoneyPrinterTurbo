import json
import re
from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.domain.content_plan import ContentBeat
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot

# =============================================================================
# Exceptions
# =============================================================================


class StoryboardError(Exception):
    """Base exception for Storyboard Agent operations."""

    def __init__(self, message: str, code: str = "STORYBOARD_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


class StoryboardOutputSchemaError(StoryboardError):
    """Raised when LLM output violates the expected JSON schema."""

    def __init__(self, message: str):
        super().__init__(message, code="LLM_OUTPUT_SCHEMA_ERROR")


class StoryboardOutputInvalidError(StoryboardError):
    """Raised when Storyboard proposals violate domain constraints."""

    def __init__(self, message: str):
        super().__init__(message, code="STORYBOARD_OUTPUT_INVALID")


class InvalidShotDurationError(StoryboardError):
    """Raised when planned shot durations deviate outside beat-level tolerance."""

    def __init__(self, message: str):
        super().__init__(message, code="INVALID_SHOT_DURATION")


class InvalidShotOrderError(StoryboardError):
    """Raised when shot local order within a beat is duplicate, non-sequential, or invalid."""

    def __init__(self, message: str):
        super().__init__(message, code="INVALID_SHOT_ORDER")


class ShotLimitExceededError(StoryboardError):
    """Raised when proposed shot count exceeds the allowed upper bound for a single beat."""

    def __init__(self, message: str):
        super().__init__(message, code="SHOT_LIMIT_EXCEEDED")


class InsufficientEvidenceError(StoryboardError):
    """Raised when factual/knowledge shots lack required evidence references."""

    def __init__(self, message: str):
        super().__init__(message, code="INSUFFICIENT_EVIDENCE")


class UnsupportedEvidenceReferenceError(StoryboardError):
    """Raised when shot cites an evidence reference not present in available context."""

    def __init__(self, message: str):
        super().__init__(message, code="UNSUPPORTED_EVIDENCE_REFERENCE")


# =============================================================================
# Storyboard Data Contracts
# =============================================================================


class StoryboardPlanningInput(BaseModel):
    """
    Input specification for the Storyboard Agent to plan shots for one ContentBeat.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    content_plan_revision_id: str = Field(
        description="ID of the ContentPlanRevision owning this beat"
    )
    beat: ContentBeat = Field(description="The ContentBeat to convert into 1..N shots")
    video_title: str | None = None
    target_aspect_ratio: str | None = Field(
        default=None, description="e.g. '16:9' or '9:16'"
    )
    user_instruction: str | None = None
    available_evidence_ids: tuple[str, ...] = Field(
        default=(),
        description="Allowed evidence IDs available in knowledge context",
    )
    language: str | None = None
    source_grounded: bool = Field(
        default=True,
        description="If True, factual shots require verified evidence citations",
    )


class StoryboardShotProposal(BaseModel):
    """
    Candidate shot proposal produced by the LLM.
    Does NOT contain authoritative system IDs (shot_id, shot_revision_id, beat_lineage_id).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    planner_ref: str = Field(description="Temporary local ref, e.g. 's1', 's2'")
    local_order: int = Field(ge=1, description="1-based local order within this beat")
    narration: str = Field(description="Spoken narration text for this shot")
    target_duration: float = Field(gt=0, description="Duration in seconds")
    visual_goal: str = Field(description="Communication purpose of this visual")
    visual_type: VisualType = Field(description="Preferred visual expression type")
    scene_description: str = Field(description="Visual description of scene/action")
    generation_prompt: str = Field(
        description="Visual prompt or query for asset synthesis/retrieval"
    )
    camera_movement: str = Field(description="Camera framing or movement")
    evidence_refs: tuple[str, ...] = Field(default=())


class StoryboardBeatOutput(BaseModel):
    """
    Structured model output schema from the LLM Storyboard Agent for one Beat.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    beat_ref: str = Field(description="Reference to the beat being converted")
    shots: tuple[StoryboardShotProposal, ...] = Field(default=())


class StoryboardExecutionResult(BaseModel):
    """
    Domain result of Storyboard planning for a Beat, containing generated Shots and ShotRevisions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    shots: tuple[Shot, ...] = Field(default=())
    shot_revisions: tuple[ShotRevision, ...] = Field(default=())


# =============================================================================
# Helper Utilities
# =============================================================================


def _strip_markdown_fences(text: str) -> str:
    """Removes surrounding markdown code fences like ```json ... ``` from LLM text."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def build_storyboard_prompt(input_data: StoryboardPlanningInput) -> str:
    """
    Constructs the focused prompt for the Storyboard Agent LLM.
    """
    beat = input_data.beat
    allowed_types = ", ".join(vt.value for vt in VisualType)

    evidence_section = ""
    if input_data.available_evidence_ids:
        ids_str = ", ".join(input_data.available_evidence_ids)
        evidence_section = f"\nAllowed Evidence IDs: [{ids_str}]\n"

    aspect_str = (
        f"\nTarget Aspect Ratio: {input_data.target_aspect_ratio}"
        if input_data.target_aspect_ratio
        else ""
    )
    lang_str = f"\nLanguage: {input_data.language}" if input_data.language else ""
    user_inst = (
        f"\nUser Style/Instructions: {input_data.user_instruction}"
        if input_data.user_instruction
        else ""
    )

    return f"""# Role: Expert Knowledge Video Storyboard Agent

## Objective:
Plan visual shots and spoken narration for ONE Content Beat in a knowledge video.
Video Title: {input_data.video_title or "Untitled"}
Plan Revision ID: {input_data.content_plan_revision_id}{aspect_str}{lang_str}{user_inst}

## Target Content Beat:
- Beat Order: {beat.order}
- Beat Type: {beat.beat_type.value}
- Intent: {beat.intent}
- Target Duration: {beat.target_duration} seconds
- Beat Evidence Refs: {list(beat.evidence_refs)}{evidence_section}
## Strict Rules:
1. One Primary Visual Per Shot:
   - Each Shot represents ONE primary visual scene. If multiple visual moments are needed, create multiple shots (1..N).
   - Do NOT introduce sub-timelines, sequential multiple cuts, or video-editing tracks within a single Shot.
2. Allowed Visual Types: [{allowed_types}].
   - Do NOT choose providers (no Runway, Pexels, Midjourney, etc.). Only select the semantic visual type.
3. Narration:
   - Spoken narration is the source of truth for audio. Keep it concise, natural, and directly tied to the visual.
   - Narration must not be empty.
4. Evidence-First Rule:
   - Factual assertions in narration must only cite available evidence IDs.
   - Do NOT hallucinate unsupported statistics, figures, or claims.
5. Duration Matching:
   - The sum of target_duration across all proposed shots must be reasonably close to {beat.target_duration}s.
6. Do NOT generate authoritative IDs:
   - Use simple strings for planner_ref (e.g. "s1", "s2").
   - Do NOT generate shot_id, shot_revision_id, or beat_lineage_id.
7. Return ONLY a single raw JSON object matching this schema:
{{
  "beat_ref": "{beat.beat_id}",
  "shots": [
    {{
      "planner_ref": "s1",
      "local_order": 1,
      "narration": "Concise spoken voiceover text.",
      "target_duration": {beat.target_duration},
      "visual_goal": "Visual purpose of the shot",
      "visual_type": "STOCK_VIDEO",
      "scene_description": "Concrete visual scene description",
      "generation_prompt": "Prompt for asset synthesis or retrieval",
      "camera_movement": "Camera framing and motion",
      "evidence_refs": []
    }}
  ]
}}
""".strip()


# =============================================================================
# Storyboard Agent Service
# =============================================================================


class StoryboardAgent:
    """
    Storyboard Agent converts a ContentBeat into 1..N validated Shot and ShotRevision entities.
    """

    def __init__(
        self,
        shot_repository: Any | None = None,
        storyboard_repository: Any | None = None,
        llm_caller: Callable[[str], str] | None = None,
        max_retries: int = 2,
        duration_tolerance_ratio: float = 0.20,
        max_shots_per_beat: int = 6,
    ):
        self._shot_repository = shot_repository
        self._storyboard_repository = storyboard_repository
        self._llm_caller = llm_caller
        self._max_retries = max_retries
        self._duration_tolerance_ratio = duration_tolerance_ratio
        self._max_shots_per_beat = max_shots_per_beat

    def _call_llm(self, prompt: str) -> str:
        if self._llm_caller is not None:
            return self._llm_caller(prompt)

        # Lazy load MoneyPrinterTurbo LLM service
        from app.services.llm import _generate_response

        return _generate_response(prompt)

    def _parse_and_validate_output(
        self,
        raw_response: str,
        input_data: StoryboardPlanningInput,
    ) -> StoryboardExecutionResult:
        cleaned = _strip_markdown_fences(raw_response)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    raise StoryboardOutputSchemaError(
                        f"Failed to parse LLM response as JSON: {exc}"
                    ) from exc
            else:
                raise StoryboardOutputSchemaError(
                    f"Failed to parse LLM response as JSON: {exc}"
                ) from exc

        try:
            proposal_output = StoryboardBeatOutput.model_validate(data)
        except ValidationError as exc:
            raise StoryboardOutputSchemaError(
                f"LLM output violated StoryboardBeatOutput schema: {exc}"
            ) from exc

        if not proposal_output.shots:
            raise StoryboardOutputInvalidError(
                "Storyboard Agent proposed zero shots; each beat must produce at least one shot."
            )

        # 1. Shot Count Bound
        if len(proposal_output.shots) > self._max_shots_per_beat:
            raise ShotLimitExceededError(
                f"Proposed shot count ({len(proposal_output.shots)}) exceeds maximum allowed "
                f"shots per beat ({self._max_shots_per_beat})."
            )

        # 2. Local Order Validation
        sorted_proposals = sorted(proposal_output.shots, key=lambda s: s.local_order)
        expected_orders = list(range(1, len(sorted_proposals) + 1))
        actual_orders = [s.local_order for s in sorted_proposals]
        if actual_orders != expected_orders:
            raise InvalidShotOrderError(
                f"Shot local_order numbers must be sequential starting at 1. Got: {actual_orders}"
            )

        # 3. Narration Non-Empty Validation
        for s in sorted_proposals:
            if not s.narration or not s.narration.strip():
                raise StoryboardOutputInvalidError(
                    f"Shot '{s.planner_ref}' (order {s.local_order}) has empty narration. "
                    "Narration is the source of truth for audio and cannot be empty."
                )

        # 4. Duration Validation
        total_duration = sum(s.target_duration for s in sorted_proposals)
        beat_dur = input_data.beat.target_duration
        max_deviation = max(2.0, beat_dur * self._duration_tolerance_ratio)
        if abs(total_duration - beat_dur) > max_deviation:
            raise InvalidShotDurationError(
                f"Total planned shot duration ({total_duration:.1f}s) deviates from beat target "
                f"({beat_dur:.1f}s) beyond acceptable tolerance of {max_deviation:.1f}s."
            )

        # 5. Evidence Validation
        if input_data.source_grounded:
            allowed_evidence_set = set(input_data.available_evidence_ids)
            all_shot_evidence_refs = [
                ref for s in sorted_proposals for ref in s.evidence_refs
            ]

            # If beat is KNOWLEDGE, at least one evidence ref must be provided
            if (
                input_data.beat.beat_type == BeatType.KNOWLEDGE
                and not all_shot_evidence_refs
            ):
                raise InsufficientEvidenceError(
                    f"Beat (order {input_data.beat.order}, type KNOWLEDGE) requires verified "
                    "evidence citations in its shots, but none were provided."
                )

            # Any cited evidence ref must exist in available evidence IDs
            if allowed_evidence_set:
                for s in sorted_proposals:
                    for ref in s.evidence_refs:
                        if ref not in allowed_evidence_set:
                            raise UnsupportedEvidenceReferenceError(
                                f"Shot '{s.planner_ref}' cites evidence '{ref}' which is not in "
                                f"allowed evidence IDs: {sorted(allowed_evidence_set)}"
                            )

        # 6. Construct Authoritative Domain Objects
        domain_shots: list[Shot] = []
        domain_revisions: list[ShotRevision] = []

        for s in sorted_proposals:
            system_shot_id = str(uuid4())
            system_shot_rev_id = str(uuid4())

            shot_entity = Shot(
                shot_id=system_shot_id,
                beat_lineage_id=input_data.beat.beat_lineage_id,
                local_order=s.local_order,
            )
            shot_rev = ShotRevision(
                shot_revision_id=system_shot_rev_id,
                shot_id=system_shot_id,
                revision_number=1,
                beat_lineage_id=input_data.beat.beat_lineage_id,
                created_from_beat_instance_id=input_data.beat.beat_id,
                narration=s.narration.strip(),
                target_duration=s.target_duration,
                visual_goal=s.visual_goal,
                visual_type=s.visual_type,
                scene_description=s.scene_description,
                generation_prompt=s.generation_prompt,
                camera_movement=s.camera_movement,
                evidence_refs=tuple(s.evidence_refs),
            )
            domain_shots.append(shot_entity)
            domain_revisions.append(shot_rev)

        return StoryboardExecutionResult(
            shots=tuple(domain_shots),
            shot_revisions=tuple(domain_revisions),
        )

    def generate_shots_for_beat(
        self,
        input_data: StoryboardPlanningInput,
        shot_repository: Any | None = None,
    ) -> StoryboardExecutionResult:
        """
        Plans and validates shots for a single ContentBeat with bounded error repair.
        """
        repo = shot_repository or self._shot_repository
        current_prompt = build_storyboard_prompt(input_data)
        last_error: Exception | None = None

        for attempt in range(1 + self._max_retries):
            raw_response = self._call_llm(current_prompt)
            try:
                result = self._parse_and_validate_output(raw_response, input_data)
                if repo is not None:
                    for shot, rev in zip(
                        result.shots, result.shot_revisions, strict=True
                    ):
                        repo.add_shot(shot)
                        repo.add_revision(rev)
                return result
            except (
                StoryboardOutputSchemaError,
                StoryboardOutputInvalidError,
                InvalidShotDurationError,
                InvalidShotOrderError,
                ShotLimitExceededError,
                InsufficientEvidenceError,
                UnsupportedEvidenceReferenceError,
            ) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    current_prompt += (
                        f"\n\n[Correction Notice]: Your previous response had validation errors:\n"
                        f"{type(exc).__name__}: {exc!s}\n"
                        f"Please fix the error and output valid JSON following the schema."
                    )

        assert last_error is not None
        raise last_error

    def plan_beat(
        self,
        input_data: StoryboardPlanningInput,
        shot_repository: Any | None = None,
    ) -> StoryboardExecutionResult:
        """Alias for generate_shots_for_beat for beat-level planning."""
        return self.generate_shots_for_beat(input_data, shot_repository=shot_repository)

    def create_draft_snapshot(
        self,
        content_plan_revision_id: str,
        shot_revisions: Sequence[ShotRevision],
        storyboard_repository: Any | None = None,
    ) -> StoryboardSnapshot:
        """
        Creates a DRAFT StoryboardSnapshot freezing exact ShotRevision IDs.
        """
        repo = storyboard_repository or self._storyboard_repository
        exact_ids = tuple(r.shot_revision_id for r in shot_revisions)
        snapshot = StoryboardSnapshot(
            storyboard_snapshot_id=str(uuid4()),
            content_plan_revision_id=content_plan_revision_id,
            shot_revision_ids=exact_ids,
            snapshot_state=StoryboardSnapshotState.DRAFT,
        )
        if repo is not None:
            repo.add_snapshot(snapshot)
        return snapshot
