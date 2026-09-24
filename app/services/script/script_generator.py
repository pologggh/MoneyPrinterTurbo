from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType
from app.domain.evidence import EvidenceItem
from app.domain.script import ScriptRevision, ScriptSegment


class ScriptGenerationError(Exception):
    """Base error for Script generation operations."""

    def __init__(self, message: str, code: str = "SCRIPT_GENERATION_ERROR") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ScriptOutputSchemaError(ScriptGenerationError):
    """Raised when LLM output violates expected JSON structure or cannot be decoded."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="LLM_OUTPUT_SCHEMA_ERROR")


class ScriptOutputInvalidError(ScriptGenerationError):
    """Raised when LLM output violates domain invariants such as beat mapping or order."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="SCRIPT_OUTPUT_INVALID")


class ScriptNeedsEvidenceError(ScriptGenerationError):
    """Raised when script narration lacks required evidence or expands evidence scope."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="NEEDS_EVIDENCE")


class ScriptSegmentProposal(BaseModel):
    """Schema for individual segment proposal emitted by LLM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content_beat_id: str
    order: int = Field(ge=1)
    narration_text: str = Field(min_length=1)
    target_duration: float = Field(gt=0)
    evidence_refs: list[str] = Field(default_factory=list)


class ScriptOutputProposal(BaseModel):
    """Schema for complete script output emitted by LLM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segments: list[ScriptSegmentProposal]


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


class ScriptGenerator:
    """
    Evidence-grounded script generator.
    Maps ContentPlanRevision beats 1:1 into immutable ScriptSegments,
    ensuring all factual narration is strictly grounded in the beat's evidence scope.
    """

    def __init__(
        self,
        llm_caller: Callable[[str], str] | None = None,
        duration_tolerance_ratio: float = 0.25,
    ) -> None:
        self.llm_caller = llm_caller
        self.duration_tolerance_ratio = duration_tolerance_ratio

    def _call_llm(self, prompt: str) -> str:
        if self.llm_caller is not None:
            return self.llm_caller(prompt)

        from app.services.llm import _generate_response

        return _generate_response(prompt)

    def _build_prompt(
        self,
        plan: ContentPlanRevision,
        evidence_by_id: dict[str, EvidenceItem],
        topic: str,
        target_duration: float,
        language: str = "zh",
        user_instruction: str | None = None,
    ) -> str:
        beat_blocks: list[str] = []
        for b in sorted(plan.beats, key=lambda beat: beat.order):
            evidence_lines: list[str] = []
            for ref in b.evidence_refs:
                ev = evidence_by_id.get(ref)
                if ev:
                    fact = ev.normalized_fact or ev.original_excerpt
                    evidence_lines.append(f"    - [{ref}] {fact}")
                else:
                    evidence_lines.append(f"    - [{ref}] (Unresolved Evidence)")

            ev_text = (
                "\n".join(evidence_lines)
                if evidence_lines
                else "    (No factual evidence attached; rhetorical/structural beat only)"
            )

            beat_blocks.append(
                f"- Beat {b.order}:\n"
                f"  content_beat_id: \"{b.beat_id}\"\n"
                f"  beat_type: {b.beat_type.value}\n"
                f"  intent: {b.intent}\n"
                f"  target_duration: {b.target_duration}s\n"
                f"  available_evidence:\n{ev_text}"
            )

        beats_str = "\n\n".join(beat_blocks)
        user_inst_str = (
            f"\nUser Instruction: {user_instruction}\n" if user_instruction else ""
        )

        return f"""You are a professional video script writer for an authoritative knowledge video.
Topic: {topic}
Overall Target Duration: {target_duration}s
Language: {language}{user_inst_str}

Content Beats to Script:
{beats_str}

STRICT INSTRUCTIONS:
1. Preserve Beat Order: Generate exactly one narration segment for each beat in sequential order (1..N).
2. Grounding Invariant:
   - For factual/knowledge beats (e.g. KNOWLEDGE), the spoken narration MUST be strictly supported by the available evidence listed for that beat.
   - Set "evidence_refs" to the list of evidence IDs actually cited.
   - You CANNOT introduce new factual claims not present in the available evidence.
   - You CANNOT cite evidence IDs outside the available evidence of that beat.
3. Rhetorical Content:
   - HOOK, TRANSITION, or SUMMARY beats may include rhetorical phrasing without evidence, provided they introduce NO unsupported factual claims.
4. Spoken Narration:
   - Write clear, engaging voiceover text suitable for speech synthesis.
   - Avoid stage directions, markdown, brackets, or parenthetical cues in narration_text.
5. Duration:
   - Match the target_duration for each segment approximately.

Return ONLY a single valid JSON object strictly matching this schema:
{{
  "segments": [
    {{
      "content_beat_id": "<exact beat_id from prompt>",
      "order": 1,
      "narration_text": "Spoken voiceover narration...",
      "target_duration": 5.0,
      "evidence_refs": ["ev_1"]
    }}
  ]
}}
""".strip()

    def generate_script(
        self,
        task_id: str,
        plan: ContentPlanRevision,
        evidence_by_id: dict[str, EvidenceItem],
        topic: str,
        target_duration: float,
        language: str = "zh",
        user_instruction: str | None = None,
        revision_number: int = 1,
        script_revision_id: str | None = None,
    ) -> ScriptRevision:
        """
        Generates and validates an immutable ScriptRevision grounded in ContentPlanRevision and Evidence.
        """
        if not plan.beats:
            raise ScriptOutputInvalidError("ContentPlanRevision contains zero beats.")

        prompt = self._build_prompt(
            plan=plan,
            evidence_by_id=evidence_by_id,
            topic=topic,
            target_duration=target_duration,
            language=language,
            user_instruction=user_instruction,
        )

        raw_response = self._call_llm(prompt)
        cleaned = _strip_markdown_fences(raw_response)

        data: dict[str, Any]
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    raise ScriptOutputSchemaError(
                        f"LLM output could not be decoded as JSON: {exc}"
                    ) from exc
            else:
                raise ScriptOutputSchemaError("LLM response did not contain a valid JSON object.")

        try:
            proposal = ScriptOutputProposal.model_validate(data)
        except ValidationError as exc:
            raise ScriptOutputSchemaError(f"LLM output schema validation failed: {exc}") from exc

        # ---------------------------------------------------------------------
        # Structural & Beat Alignment Validation
        # ---------------------------------------------------------------------
        beats_by_id = {b.beat_id: b for b in plan.beats}
        beats_by_order = {b.order: b for b in plan.beats}

        if len(proposal.segments) != len(plan.beats):
            raise ScriptOutputInvalidError(
                f"Segment count mismatch: expected {len(plan.beats)} segments (1 per beat), "
                f"got {len(proposal.segments)}."
            )

        # Check ordering and beat mapping
        sorted_proposals = sorted(proposal.segments, key=lambda s: s.order)
        expected_orders = list(range(1, len(sorted_proposals) + 1))
        actual_orders = [s.order for s in sorted_proposals]
        if actual_orders != expected_orders:
            raise ScriptOutputInvalidError(
                f"Segment order numbers must be sequential starting at 1. Got: {actual_orders}"
            )

        rev_id = script_revision_id or str(uuid4())
        created_segments: list[ScriptSegment] = []

        for prop in sorted_proposals:
            beat = beats_by_id.get(prop.content_beat_id)
            if beat is None:
                raise ScriptOutputInvalidError(
                    f"Segment order {prop.order} references unknown content_beat_id '{prop.content_beat_id}'."
                )

            if beat.order != prop.order:
                raise ScriptOutputInvalidError(
                    f"Segment order {prop.order} does not match originating beat order {beat.order}."
                )

            # -----------------------------------------------------------------
            # Evidence Grounding Invariant
            # -----------------------------------------------------------------
            beat_allowed_evidence = set(beat.evidence_refs)
            segment_evidence_set = set(prop.evidence_refs)

            # Subset Rule: Segment evidence refs must be subset of Beat evidence refs
            unsupported_refs = segment_evidence_set - beat_allowed_evidence
            if unsupported_refs:
                raise ScriptNeedsEvidenceError(
                    f"Segment order {prop.order} cited evidence refs {sorted(unsupported_refs)} "
                    f"which were not in beat '{beat.beat_id}' evidence scope."
                )

            # All cited evidence must resolve to an actual EvidenceItem
            for ref in prop.evidence_refs:
                if ref not in evidence_by_id:
                    raise ScriptNeedsEvidenceError(
                        f"Segment order {prop.order} cites evidence ref '{ref}' "
                        f"which does not exist in the EvidenceSnapshot."
                    )

            # Factual requirement: KNOWLEDGE beat segments must have evidence grounding
            if beat.beat_type == BeatType.KNOWLEDGE and not prop.evidence_refs:
                raise ScriptNeedsEvidenceError(
                    f"Factual KNOWLEDGE beat (order {beat.order}, id '{beat.beat_id}') "
                    "requires evidence references in narration, but none were cited."
                )

            seg = ScriptSegment(
                script_revision_id=rev_id,
                content_beat_id=beat.beat_id,
                beat_lineage_id=beat.beat_lineage_id,
                order=prop.order,
                narration_text=prop.narration_text.strip(),
                target_duration=prop.target_duration,
                beat_type=beat.beat_type,
                evidence_refs=tuple(prop.evidence_refs),
            )
            created_segments.append(seg)

        # ---------------------------------------------------------------------
        # Duration Tolerance Validation
        # ---------------------------------------------------------------------
        total_duration = sum(s.target_duration for s in created_segments)
        max_deviation = max(5.0, target_duration * self.duration_tolerance_ratio)
        if abs(total_duration - target_duration) > max_deviation:
            raise ScriptOutputInvalidError(
                f"Total script duration ({total_duration:.1f}s) deviates from target "
                f"({target_duration:.1f}s) beyond acceptable tolerance of {max_deviation:.1f}s."
            )

        return ScriptRevision.create(
            script_revision_id=rev_id,
            task_id=task_id,
            content_plan_revision_id=plan.content_plan_revision_id,
            segments=created_segments,
            overall_target_duration=round(total_duration, 2),
            language=language,
            revision_number=revision_number,
        )
