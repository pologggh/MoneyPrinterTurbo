import json
import re
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType
from app.domain.plan_diff import generate_beat_lineage_id

# =============================================================================
# Exceptions
# =============================================================================


class PlannerError(Exception):
    """Base exception for Content Planner operations."""

    def __init__(self, message: str, code: str = "PLANNER_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


class PlannerOutputSchemaError(PlannerError):
    """Raised when LLM output cannot be parsed into the expected JSON schema."""

    def __init__(self, message: str):
        super().__init__(message, code="LLM_OUTPUT_SCHEMA_ERROR")


class PlannerProviderError(PlannerError):
    """Raised when the LLM provider request fails before producing planner output."""

    def __init__(self, message: str):
        super().__init__(message, code="LLM_PROVIDER_ERROR")


class PlannerOutputInvalidError(PlannerError):
    """Raised when LLM output violates domain constraints or semantic invariants."""

    def __init__(self, message: str):
        super().__init__(message, code="PLANNER_OUTPUT_INVALID")


class InsufficientEvidenceError(PlannerError):
    """Raised when factual/knowledge beats lack required source evidence."""

    def __init__(self, message: str):
        super().__init__(message, code="INSUFFICIENT_EVIDENCE")


class InvalidBeatLineageInheritanceError(PlannerError):
    """Raised when beat lineage inheritance from a previous plan is invalid or conflicting."""

    def __init__(self, message: str):
        super().__init__(message, code="INVALID_BEAT_LINEAGE_INHERITANCE")


class InvalidDurationPlanError(PlannerError):
    """Raised when planned beat durations deviate too far from target video duration."""

    def __init__(self, message: str):
        super().__init__(message, code="INVALID_DURATION_PLAN")


# =============================================================================
# Planner Data Contracts
# =============================================================================


class PlannerInput(BaseModel):
    """
    Input specification for the Content Planner.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str
    target_video_duration: float = Field(
        gt=0, description="Target total video duration in seconds"
    )
    title: str | None = None
    user_instruction: str | None = None
    knowledge_context: tuple[str, ...] = Field(
        default=(),
        description="Source snippets, notes, or evidence identifiers available to the planner",
    )
    available_evidence_ids: tuple[str, ...] = Field(
        default=(),
        description="Explicit allowed evidence reference IDs if defined in context",
    )
    previous_content_plan_revision_id: str | None = None
    source_grounded: bool = Field(
        default=False,
        description="If True, factual content (KNOWLEDGE) strictly requires evidence references",
    )
    global_retrieval_snapshot_id: str | None = Field(
        default=None,
        description="Optional reference to the retrieval snapshot used during evidence collection",
    )


class PlannerBeatProposal(BaseModel):
    """
    Candidate beat proposal produced by the LLM in structured output.
    Does NOT contain authoritative system IDs (beat_id, beat_lineage_id).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    planner_ref: str = Field(description="Temporary local ref, e.g. 'b1', 'b2'")
    beat_type: BeatType
    intent: str
    order: int = Field(ge=1)
    target_duration: float = Field(gt=0)
    importance: float = Field(ge=0.0, le=1.0)
    evidence_refs: tuple[str, ...] = Field(default=())
    inherit_from: str | None = Field(
        default=None,
        description="Reference to previous beat's beat_id or beat_lineage_id when revising",
    )


class PlannerOutput(BaseModel):
    """
    Structured model output schema from the LLM Content Planner.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    beats: tuple[PlannerBeatProposal, ...] = Field(default=())
    planner_notes: str | None = None


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


def build_planner_prompt(
    input_data: PlannerInput,
    previous_plan: ContentPlanRevision | None = None,
) -> str:
    """
    Constructs the focused prompt for the Content Planner LLM.
    """
    allowed_types = ", ".join(bt.value for bt in BeatType)

    knowledge_section = ""
    if input_data.knowledge_context:
        snippets = "\n".join(f"- {s}" for s in input_data.knowledge_context)
        knowledge_section = f"""
## Available Knowledge Context (Evidence):
{snippets}
"""
    if input_data.available_evidence_ids:
        ids_str = ", ".join(input_data.available_evidence_ids)
        knowledge_section += f"\nAllowed Evidence IDs: [{ids_str}]\n"

    revision_section = ""
    if previous_plan is not None:
        beats_summary = []
        for b in sorted(previous_plan.beats, key=lambda x: x.order):
            beats_summary.append(
                f"- Order {b.order} | BeatID: '{b.beat_id}' | LineageID: '{b.beat_lineage_id}' | "
                f"Type: {b.beat_type.value} | Intent: {b.intent} | Duration: {b.target_duration}s"
            )
        revision_section = f"""
## Previous Content Plan Revision (Revision #{previous_plan.revision_number}):
Plan ID: {previous_plan.content_plan_revision_id}
Topic: {previous_plan.topic}
Previous Beats:
{chr(10).join(beats_summary)}

When revising, if a new beat corresponds to a previous beat's semantic lineage, specify:
"inherit_from": "<BeatID or LineageID from above>"
If a beat is brand new, specify:
"inherit_from": null
"""

    instruction_text = (
        f"\nUser Additional Instructions: {input_data.user_instruction}\n"
        if input_data.user_instruction
        else ""
    )

    return f"""# Role: Expert Knowledge Video Content Planner

## Objective:
Plan the high-level narrative structure (Content Beats) for a short knowledge/science video.
Topic: {input_data.topic}
Target Video Duration: {input_data.target_video_duration} seconds{instruction_text}
{knowledge_section}
{revision_section}
## Strict Rules:
1. Plan WHAT the video communicates at each beat. Do NOT decide exact camera shots or visual assets.
2. Allowed Beat Types: [{allowed_types}].
3. Evidence-First Rule:
   - KNOWLEDGE beats MUST cite valid evidence reference IDs from the Knowledge Context.
   - EXAMPLE beats that assert factual or empirical data MUST cite evidence references.
   - HOOK and TRANSITION beats do not require evidence.
   - Do NOT invent unsupported facts.
4. Duration Planning:
   - The sum of target_duration across all beats must be approximately {input_data.target_video_duration}s.
5. Do NOT generate authoritative IDs:
   - Use simple strings for planner_ref (e.g. "b1", "b2").
   - Do NOT generate beat_instance_id, beat_lineage_id, or content_plan_revision_id.
6. Return ONLY a single raw JSON object matching this schema:
{{
  "title": "{input_data.title or input_data.topic}",
  "beats": [
    {{
      "planner_ref": "b1",
      "beat_type": "HOOK",
      "intent": "Hook the viewer with a surprising problem statement",
      "order": 1,
      "target_duration": 4.0,
      "importance": 0.9,
      "evidence_refs": [],
      "inherit_from": null
    }}
  ],
  "planner_notes": "optional brief notes"
}}
""".strip()


# =============================================================================
# Content Planner Implementation
# =============================================================================


class ContentPlanner:
    """
    Content Planner converts user topic and knowledge context into a validated
    ContentPlanRevision with ordered ContentBeats.
    """

    def __init__(
        self,
        repository: Any | None = None,
        llm_caller: Callable[[str], str] | None = None,
        max_retries: int = 2,
        duration_tolerance_ratio: float = 0.15,
    ):
        self._repository = repository
        self._llm_caller = llm_caller
        self._max_retries = max_retries
        self._duration_tolerance_ratio = duration_tolerance_ratio

    def _call_llm(self, prompt: str) -> str:
        if self._llm_caller is not None:
            return self._llm_caller(prompt)

        # Lazy load MoneyPrinterTurbo LLM service to avoid unnecessary dependencies during domain unit tests
        from app.services.llm import _generate_response

        return _generate_response(prompt)

    def _parse_and_validate_output(
        self,
        raw_response: str,
        input_data: PlannerInput,
        previous_plan: ContentPlanRevision | None,
    ) -> ContentPlanRevision:
        cleaned = _strip_markdown_fences(raw_response)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Fallback regex extraction for json object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    raise PlannerOutputSchemaError(
                        f"Failed to parse LLM response as JSON: {exc}"
                    ) from exc
            else:
                raise PlannerOutputSchemaError(
                    f"Failed to parse LLM response as JSON: {exc}"
                ) from exc

        try:
            proposal_output = PlannerOutput.model_validate(data)
        except ValidationError as exc:
            raise PlannerOutputSchemaError(
                f"LLM output violated PlannerOutput schema: {exc}"
            ) from exc

        if not proposal_output.beats:
            raise PlannerOutputInvalidError(
                "Planner proposed zero beats; a plan must have at least one beat."
            )

        # 1. Order Validation
        sorted_proposals = sorted(proposal_output.beats, key=lambda b: b.order)
        expected_orders = list(range(1, len(sorted_proposals) + 1))
        actual_orders = [b.order for b in sorted_proposals]
        if actual_orders != expected_orders:
            raise PlannerOutputInvalidError(
                f"Beat order numbers must be sequential starting at 1. Got: {actual_orders}"
            )

        # 2. Duration Validation
        total_duration = sum(b.target_duration for b in sorted_proposals)
        target_dur = input_data.target_video_duration
        max_deviation = max(5.0, target_dur * self._duration_tolerance_ratio)
        if abs(total_duration - target_dur) > max_deviation:
            raise InvalidDurationPlanError(
                f"Total planned duration ({total_duration:.1f}s) deviates from target "
                f"({target_dur:.1f}s) beyond acceptable tolerance of {max_deviation:.1f}s."
            )

        # 3. Evidence Validation (Evidence-First)
        if input_data.source_grounded:
            allowed_evidence_set = set(input_data.available_evidence_ids)

            # First verify KNOWLEDGE beats have evidence references
            for p in sorted_proposals:
                if p.beat_type == BeatType.KNOWLEDGE and not p.evidence_refs:
                    raise InsufficientEvidenceError(
                        f"KNOWLEDGE beat (order {p.order}, ref '{p.planner_ref}') "
                        "requires evidence references, but none were provided."
                    )

            # Second ensure allowed evidence IDs are present
            if not allowed_evidence_set:
                raise InsufficientEvidenceError(
                    "source_grounded is True but no available evidence IDs were provided to ground beats."
                )

            # Third verify all evidence references are grounded
            for p in sorted_proposals:
                for ref in p.evidence_refs:
                    if ref not in allowed_evidence_set:
                        raise InsufficientEvidenceError(
                            f"Evidence reference '{ref}' in beat {p.order} is not "
                            f"grounded in available evidence IDs: {sorted(allowed_evidence_set)}"
                        )

        # 4. Lineage and Inheritance Validation
        claimed_lineages: set[str] = set()
        old_beats_by_id = (
            {b.beat_id: b for b in previous_plan.beats} if previous_plan else {}
        )
        old_beats_by_lineage = (
            {b.beat_lineage_id: b for b in previous_plan.beats} if previous_plan else {}
        )

        domain_beats: list[ContentBeat] = []
        for p in sorted_proposals:
            if p.inherit_from:
                if previous_plan is None:
                    raise InvalidBeatLineageInheritanceError(
                        f"Beat '{p.planner_ref}' specifies inherit_from '{p.inherit_from}', "
                        "but no previous plan was supplied."
                    )
                # Match against old beat_id or old beat_lineage_id
                matched_old_beat = old_beats_by_id.get(
                    p.inherit_from
                ) or old_beats_by_lineage.get(p.inherit_from)
                if matched_old_beat is None:
                    raise InvalidBeatLineageInheritanceError(
                        f"inherit_from reference '{p.inherit_from}' does not match any beat "
                        f"in previous plan revision '{previous_plan.content_plan_revision_id}'."
                    )

                lineage_id = matched_old_beat.beat_lineage_id
                if lineage_id in claimed_lineages:
                    raise InvalidBeatLineageInheritanceError(
                        f"Duplicate inheritance: multiple proposals inherit from the same "
                        f"previous beat lineage '{lineage_id}'."
                    )
                claimed_lineages.add(lineage_id)
            else:
                # Brand new lineage generated by the system
                lineage_id = generate_beat_lineage_id()

            domain_beats.append(
                ContentBeat(
                    beat_id=str(uuid4()),
                    beat_lineage_id=lineage_id,
                    beat_type=p.beat_type,
                    order=p.order,
                    intent=p.intent,
                    target_duration=p.target_duration,
                    importance=p.importance,
                    evidence_refs=tuple(p.evidence_refs),
                )
            )

        revision_number = (previous_plan.revision_number + 1) if previous_plan else 1
        new_plan = ContentPlanRevision(
            content_plan_revision_id=str(uuid4()),
            revision_number=revision_number,
            topic=proposal_output.title or input_data.title or input_data.topic,
            overall_target_duration=input_data.target_video_duration,
            beats=tuple(domain_beats),
            global_retrieval_snapshot_id=input_data.global_retrieval_snapshot_id,
        )

        return new_plan

    def plan(
        self,
        input_data: PlannerInput,
        repository: Any | None = None,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> ContentPlanRevision:
        """
        Executes the content planning workflow with deterministic validation and bounded retries.
        """
        repo = repository or self._repository
        previous_plan: ContentPlanRevision | None = None
        if input_data.previous_content_plan_revision_id:
            if repo is None:
                raise PlannerOutputInvalidError(
                    "previous_content_plan_revision_id specified, but no repository was "
                    "provided to load the previous plan."
                )
            previous_plan = repo.get_revision(
                input_data.previous_content_plan_revision_id
            )
            if previous_plan is None:
                raise PlannerOutputInvalidError(
                    f"Previous ContentPlanRevision '{input_data.previous_content_plan_revision_id}' "
                    "not found in repository."
                )

        start_ev = None
        if trace_writer is not None and trace_context is not None:
            from app.domain.trace import PlannerStartedTraceData, TraceEventType

            start_attrs = PlannerStartedTraceData(
                topic=input_data.topic,
                target_duration=input_data.target_video_duration,
                planner_version="v1",
            )
            start_ev = trace_writer.start_event(
                context=trace_context,
                event_type=TraceEventType.CONTENT_PLANNER_STARTED,
                attributes=start_attrs,
            )

        current_prompt = build_planner_prompt(input_data, previous_plan)
        last_error: Exception | None = None

        for attempt in range(1 + self._max_retries):
            raw_response = self._call_llm(current_prompt)
            if raw_response.startswith("Error:"):
                provider_error = raw_response.removeprefix("Error:").strip()
                raise PlannerProviderError(
                    f"LLM provider request failed: {provider_error or 'unknown error'}"
                )
            try:
                plan_revision = self._parse_and_validate_output(
                    raw_response, input_data, previous_plan
                )
                if repo is not None:
                    repo.add_revision(plan_revision)

                if start_ev is not None and trace_writer is not None:
                    from app.domain.trace import (
                        PlannerCompletedTraceData,
                        TraceEventStatus,
                    )

                    comp_attrs = PlannerCompletedTraceData(
                        content_plan_revision_id=plan_revision.content_plan_revision_id,
                        beat_count=len(plan_revision.beats),
                        evidence_ref_count=sum(len(b.evidence_refs) for b in plan_revision.beats),
                        repair_retry_count=attempt,
                    )
                    trace_writer.complete_event(
                        event=start_ev,
                        status=TraceEventStatus.SUCCEEDED,
                        attributes_update=comp_attrs,
                    )

                return plan_revision
            except (
                PlannerOutputSchemaError,
                PlannerOutputInvalidError,
                InsufficientEvidenceError,
                InvalidBeatLineageInheritanceError,
                InvalidDurationPlanError,
            ) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    # Provide targeted correction feedback for repair attempt
                    current_prompt += (
                        f"\n\n[Correction Notice]: Your previous response had validation errors:\n"
                        f"{type(exc).__name__}: {exc!s}\n"
                        f"Please fix the error and output valid JSON following the schema."
                    )

        assert last_error is not None
        if start_ev is not None and trace_writer is not None:
            trace_writer.fail_event(
                event=start_ev,
                error_code=type(last_error).__name__,
                attributes_update={"repair_retry_count": self._max_retries},
            )
        raise last_error
