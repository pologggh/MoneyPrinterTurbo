from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import BeatType


def compute_script_fingerprint(
    content_plan_revision_id: str,
    segments: Sequence[ScriptSegment],
) -> str:
    """
    Computes a deterministic, content-based SHA-256 fingerprint for a ScriptRevision.
    Covers the source ContentPlanRevision ID, ordered segments (beat ID, order,
    narration text, target duration, and sorted evidence references).
    Excludes system IDs, database order, and timestamps.
    """
    canonical_payload = {
        "content_plan_revision_id": content_plan_revision_id,
        "segments": [
            {
                "content_beat_id": s.content_beat_id,
                "order": s.order,
                "narration_text": s.narration_text.strip(),
                "target_duration": round(float(s.target_duration), 2),
                "evidence_refs": sorted(s.evidence_refs),
            }
            for s in sorted(segments, key=lambda seg: seg.order)
        ],
    }
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ScriptSegment(BaseModel):
    """
    Immutable representation of a single narration segment derived from a ContentBeat.
    Authoritative source of text for subsequent audio/TTS synthesis.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    script_segment_id: str = Field(default_factory=lambda: str(uuid4()))
    script_revision_id: str
    content_beat_id: str = Field(description="Reference to the originating ContentBeat.beat_id")
    beat_lineage_id: str | None = Field(
        default=None, description="Lineage ID of originating ContentBeat if available"
    )
    order: int = Field(ge=1, description="1-based ordering preserving ContentBeat.order")
    narration_text: str = Field(description="Authoritative spoken narration text")
    target_duration: float = Field(gt=0, description="Estimated/target duration in seconds")
    beat_type: BeatType | None = Field(
        default=None, description="BeatType of originating ContentBeat"
    )
    evidence_refs: tuple[str, ...] = Field(
        default=(),
        description="EvidenceItem IDs grounding factual claims in this segment",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScriptRevision(BaseModel):
    """
    Immutable revision of a video script.
    Consists of ordered ScriptSegments mapped 1:1 to ContentBeats with verified evidence references.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    script_revision_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = Field(description="Associated KnowledgeVideoTask ID")
    content_plan_revision_id: str = Field(
        description="ContentPlanRevision ID that produced this script"
    )
    revision_number: int = Field(default=1, ge=1, description="1-based script revision number")
    overall_target_duration: float = Field(
        gt=0, description="Overall duration in seconds matching content plan"
    )
    language: str = Field(default="zh", description="Narration language code")
    content_fingerprint: str = Field(
        description="Deterministic SHA-256 fingerprint of plan reference and segments"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    segments: tuple[ScriptSegment, ...] = Field(
        default=(),
        description="Immutable sequence of script segments ordered by segment.order",
    )

    @classmethod
    def create(
        cls,
        task_id: str,
        content_plan_revision_id: str,
        segments: Sequence[ScriptSegment],
        overall_target_duration: float,
        language: str = "zh",
        revision_number: int = 1,
        script_revision_id: str | None = None,
        created_at: datetime | None = None,
    ) -> ScriptRevision:
        rev_id = script_revision_id or str(uuid4())
        ordered_segments = tuple(sorted(segments, key=lambda s: s.order))
        fingerprint = compute_script_fingerprint(content_plan_revision_id, ordered_segments)
        return cls(
            script_revision_id=rev_id,
            task_id=task_id,
            content_plan_revision_id=content_plan_revision_id,
            revision_number=revision_number,
            overall_target_duration=overall_target_duration,
            language=language,
            content_fingerprint=fingerprint,
            created_at=created_at or datetime.now(UTC),
            segments=ordered_segments,
        )
