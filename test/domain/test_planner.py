import json
import uuid

import pytest

from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType
from app.domain.plan_diff import diff_content_plans
from app.domain.planner import (
    ContentPlanner,
    InsufficientEvidenceError,
    InvalidBeatLineageInheritanceError,
    InvalidDurationPlanError,
    PlannerInput,
    PlannerOutputInvalidError,
    PlannerOutputSchemaError,
)


class InMemoryContentPlanRepo:
    """Lightweight in-memory repository for unit testing."""

    def __init__(self):
        self._store: dict[str, ContentPlanRevision] = {}

    def add_revision(self, plan: ContentPlanRevision) -> ContentPlanRevision:
        self._store[plan.content_plan_revision_id] = plan
        return plan

    def get_revision(self, revision_id: str) -> ContentPlanRevision | None:
        return self._store.get(revision_id)


def _make_sample_llm_json(
    title: str = "Quantum Computing Explained",
    beats: list[dict] | None = None,
) -> str:
    if beats is None:
        beats = [
            {
                "planner_ref": "b1",
                "beat_type": "HOOK",
                "intent": "Introduce the mind-bending concept of superposition",
                "order": 1,
                "target_duration": 5.0,
                "importance": 0.9,
                "evidence_refs": [],
                "inherit_from": None,
            },
            {
                "planner_ref": "b2",
                "beat_type": "KNOWLEDGE",
                "intent": "Explain qubits and quantum entanglement",
                "order": 2,
                "target_duration": 20.0,
                "importance": 1.0,
                "evidence_refs": ["ev_qubit_1"],
                "inherit_from": None,
            },
            {
                "planner_ref": "b3",
                "beat_type": "SUMMARY",
                "intent": "Summarize potential impact on cryptography and medicine",
                "order": 3,
                "target_duration": 5.0,
                "importance": 0.8,
                "evidence_refs": [],
                "inherit_from": None,
            },
        ]
    return json.dumps({
        "title": title,
        "beats": beats,
        "planner_notes": "Balanced pacing for 30s knowledge video",
    })


# 1. Valid planner output produces a valid ContentPlanRevision
def test_valid_planner_output_produces_content_plan_revision():
    valid_json = _make_sample_llm_json()
    stub_llm = lambda prompt: f"```json\n{valid_json}\n```"

    planner = ContentPlanner(llm_caller=stub_llm)
    planner_input = PlannerInput(
        topic="Quantum Computing",
        target_video_duration=30.0,
        knowledge_context=("ev_qubit_1: Qubits exploit superposition and entanglement.",),
        available_evidence_ids=("ev_qubit_1",),
    )

    plan = planner.plan(planner_input)

    assert isinstance(plan, ContentPlanRevision)
    assert plan.revision_number == 1
    assert plan.topic == "Quantum Computing Explained"
    assert plan.overall_target_duration == 30.0
    assert len(plan.beats) == 3
    assert plan.beats[0].beat_type == BeatType.HOOK
    assert plan.beats[1].beat_type == BeatType.KNOWLEDGE
    assert plan.beats[2].beat_type == BeatType.SUMMARY
    assert plan.beats[1].evidence_refs == ("ev_qubit_1",)


# 2. System generates authoritative Beat identities
def test_system_generates_authoritative_beat_identities():
    valid_json = _make_sample_llm_json()
    stub_llm = lambda prompt: valid_json

    planner = ContentPlanner(llm_caller=stub_llm)
    planner_input = PlannerInput(topic="Quantum Computing", target_video_duration=30.0)

    plan = planner.plan(planner_input)

    # Authority: system generates valid UUIDs
    uuid.UUID(plan.content_plan_revision_id)
    for beat in plan.beats:
        uuid.UUID(beat.beat_id)
        uuid.UUID(beat.beat_lineage_id)
        # Verify planner_ref was discarded and not used as authoritative ID
        assert beat.beat_id not in ("b1", "b2", "b3")
        assert beat.beat_lineage_id not in ("b1", "b2", "b3")

    # Distinct IDs for distinct beats
    beat_ids = {b.beat_id for b in plan.beats}
    lineage_ids = {b.beat_lineage_id for b in plan.beats}
    assert len(beat_ids) == len(plan.beats)
    assert len(lineage_ids) == len(plan.beats)


# 3. LLM cannot directly control beat_lineage_id
def test_llm_cannot_directly_control_beat_lineage_id():
    # Attempting to pass forbidden beat_lineage_id in JSON
    malicious_json = json.dumps({
        "title": "Quantum",
        "beats": [
            {
                "planner_ref": "b1",
                "beat_type": "HOOK",
                "intent": "Hook",
                "order": 1,
                "target_duration": 30.0,
                "importance": 0.5,
                "evidence_refs": [],
                "inherit_from": None,
                "beat_lineage_id": "malicious-lineage-12345",
            }
        ],
    })
    planner = ContentPlanner(llm_caller=lambda p: malicious_json)
    planner_input = PlannerInput(topic="Quantum", target_video_duration=30.0)

    # Pydantic extra='forbid' rejects beat_lineage_id injected into proposal
    with pytest.raises(PlannerOutputSchemaError):
        planner.plan(planner_input)


# 4. KNOWLEDGE beat without required evidence is rejected
def test_knowledge_beat_without_required_evidence_rejected():
    beats = [
        {
            "planner_ref": "b1",
            "beat_type": "KNOWLEDGE",
            "intent": "Unsubstantiated factual claim",
            "order": 1,
            "target_duration": 30.0,
            "importance": 0.9,
            "evidence_refs": [],  # Missing evidence!
            "inherit_from": None,
        }
    ]
    planner = ContentPlanner(llm_caller=lambda p: _make_sample_llm_json(beats=beats))
    planner_input = PlannerInput(
        topic="Physics",
        target_video_duration=30.0,
        source_grounded=True,
    )

    with pytest.raises(InsufficientEvidenceError) as exc_info:
        planner.plan(planner_input)
    assert "requires evidence references" in str(exc_info.value)

    # Also test citing evidence not in available_evidence_ids
    beats_unallowed_ref = [
        {
            "planner_ref": "b1",
            "beat_type": "KNOWLEDGE",
            "intent": "Factual claim with fake source",
            "order": 1,
            "target_duration": 30.0,
            "importance": 0.9,
            "evidence_refs": ["unauthorized_ref_99"],
            "inherit_from": None,
        }
    ]
    planner_unallowed = ContentPlanner(
        llm_caller=lambda p: _make_sample_llm_json(beats=beats_unallowed_ref)
    )
    planner_input_with_ids = PlannerInput(
        topic="Physics",
        target_video_duration=30.0,
        available_evidence_ids=("allowed_ref_1",),
        source_grounded=True,
    )

    with pytest.raises(InsufficientEvidenceError) as exc_info2:
        planner_unallowed.plan(planner_input_with_ids)
    assert "not grounded in available evidence IDs" in str(exc_info2.value)


# 5. Invalid beat type or schema is rejected
def test_invalid_beat_type_or_schema_rejected():
    # Case A: completely malformed JSON
    planner_bad_json = ContentPlanner(llm_caller=lambda p: "Not even JSON!")
    with pytest.raises(PlannerOutputSchemaError):
        planner_bad_json.plan(PlannerInput(topic="Math", target_video_duration=10.0))

    # Case B: invalid BeatType enum
    bad_enum_json = json.dumps({
        "title": "Math",
        "beats": [
            {
                "planner_ref": "b1",
                "beat_type": "DANCE_ROUTINE",  # Invalid BeatType
                "intent": "Dance",
                "order": 1,
                "target_duration": 10.0,
                "importance": 0.5,
                "evidence_refs": [],
                "inherit_from": None,
            }
        ],
    })
    planner_bad_enum = ContentPlanner(llm_caller=lambda p: bad_enum_json)
    with pytest.raises(PlannerOutputSchemaError):
        planner_bad_enum.plan(PlannerInput(topic="Math", target_video_duration=10.0))

    # Case C: non-sequential orders
    bad_order_json = json.dumps({
        "title": "Math",
        "beats": [
            {
                "planner_ref": "b1",
                "beat_type": "HOOK",
                "intent": "Hook",
                "order": 2,  # starts with 2 instead of 1
                "target_duration": 10.0,
                "importance": 0.5,
                "evidence_refs": [],
                "inherit_from": None,
            }
        ],
    })
    planner_bad_order = ContentPlanner(llm_caller=lambda p: bad_order_json)
    with pytest.raises(PlannerOutputInvalidError):
        planner_bad_order.plan(PlannerInput(topic="Math", target_video_duration=10.0))


# 6. Duration plan outside tolerance is rejected
def test_duration_plan_outside_tolerance_rejected():
    # Target is 60s, tolerance ratio is 0.15 (max deviation 9.0s), beats sum to 20s
    beats = [
        {
            "planner_ref": "b1",
            "beat_type": "HOOK",
            "intent": "Hook",
            "order": 1,
            "target_duration": 20.0,  # 20s << 60s
            "importance": 0.5,
            "evidence_refs": [],
            "inherit_from": None,
        }
    ]
    planner = ContentPlanner(llm_caller=lambda p: _make_sample_llm_json(beats=beats))
    planner_input = PlannerInput(topic="History", target_video_duration=60.0)

    with pytest.raises(InvalidDurationPlanError) as exc_info:
        planner.plan(planner_input)
    assert "deviates from target (60.0s) beyond acceptable tolerance" in str(exc_info.value)


# 7. Previous-plan revision with valid inherit_from preserves beat_lineage_id
def test_previous_plan_revision_with_valid_inherit_from_preserves_lineage():
    repo = InMemoryContentPlanRepo()

    old_beat_1 = ContentBeat(
        beat_id="old-beat-uuid-1",
        beat_lineage_id="lineage-uuid-1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Original hook",
        target_duration=10.0,
        importance=0.8,
    )
    old_beat_2 = ContentBeat(
        beat_id="old-beat-uuid-2",
        beat_lineage_id="lineage-uuid-2",
        beat_type=BeatType.KNOWLEDGE,
        order=2,
        intent="Original knowledge",
        target_duration=20.0,
        importance=1.0,
        evidence_refs=("ev1",),
    )
    old_rev = ContentPlanRevision(
        content_plan_revision_id="rev-uuid-1",
        revision_number=1,
        topic="Biology",
        overall_target_duration=30.0,
        beats=(old_beat_1, old_beat_2),
    )
    repo.add_revision(old_rev)

    # Revision proposes new beats, inheriting from old_beat_1 and old_beat_2
    revised_beats = [
        {
            "planner_ref": "nb1",
            "beat_type": "HOOK",
            "intent": "Refined punchy hook",
            "order": 1,
            "target_duration": 10.0,
            "importance": 0.9,
            "evidence_refs": [],
            "inherit_from": "old-beat-uuid-1",  # inherits from old beat_id
        },
        {
            "planner_ref": "nb2",
            "beat_type": "KNOWLEDGE",
            "intent": "Updated knowledge detail",
            "order": 2,
            "target_duration": 20.0,
            "importance": 1.0,
            "evidence_refs": ["ev1"],
            "inherit_from": "lineage-uuid-2",  # inherits from old lineage_id
        },
    ]
    planner = ContentPlanner(
        repository=repo,
        llm_caller=lambda p: _make_sample_llm_json(beats=revised_beats),
    )

    new_plan = planner.plan(
        PlannerInput(
            topic="Biology",
            target_video_duration=30.0,
            previous_content_plan_revision_id="rev-uuid-1",
            available_evidence_ids=("ev1",),
        )
    )

    assert new_plan.revision_number == 2
    assert new_plan.content_plan_revision_id != old_rev.content_plan_revision_id
    assert len(new_plan.beats) == 2

    # Verify lineages are preserved
    assert new_plan.beats[0].beat_lineage_id == "lineage-uuid-1"
    assert new_plan.beats[1].beat_lineage_id == "lineage-uuid-2"

    # Verify new unique beat_ids were generated
    assert new_plan.beats[0].beat_id != "old-beat-uuid-1"
    assert new_plan.beats[1].beat_id != "old-beat-uuid-2"


# 8. Invalid inherit_from reference is rejected
def test_invalid_inherit_from_reference_rejected():
    repo = InMemoryContentPlanRepo()
    old_rev = ContentPlanRevision(
        content_plan_revision_id="rev-uuid-1",
        revision_number=1,
        topic="Physics",
        overall_target_duration=30.0,
        beats=(
            ContentBeat(
                beat_id="b1",
                beat_lineage_id="l1",
                beat_type=BeatType.HOOK,
                order=1,
                intent="Hook",
                target_duration=30.0,
                importance=0.5,
            ),
        ),
    )
    repo.add_revision(old_rev)

    # inherit_from points to non-existent ID
    bad_inherit_beats = [
        {
            "planner_ref": "b1",
            "beat_type": "HOOK",
            "intent": "Hook",
            "order": 1,
            "target_duration": 30.0,
            "importance": 0.5,
            "evidence_refs": [],
            "inherit_from": "ghost-beat-id",
        }
    ]
    planner = ContentPlanner(
        repository=repo,
        llm_caller=lambda p: _make_sample_llm_json(beats=bad_inherit_beats),
    )

    with pytest.raises(InvalidBeatLineageInheritanceError) as exc_info:
        planner.plan(
            PlannerInput(
                topic="Physics",
                target_video_duration=30.0,
                previous_content_plan_revision_id="rev-uuid-1",
            )
        )
    assert "does not match any beat in previous plan revision" in str(exc_info.value)


# 9. Duplicate inheritance to the same old Beat is rejected
def test_duplicate_inheritance_to_same_old_beat_rejected():
    repo = InMemoryContentPlanRepo()
    old_rev = ContentPlanRevision(
        content_plan_revision_id="rev-uuid-1",
        revision_number=1,
        topic="Physics",
        overall_target_duration=30.0,
        beats=(
            ContentBeat(
                beat_id="b1",
                beat_lineage_id="l1",
                beat_type=BeatType.HOOK,
                order=1,
                intent="Hook",
                target_duration=30.0,
                importance=0.5,
            ),
        ),
    )
    repo.add_revision(old_rev)

    # Both proposed beats claim inheritance from the same old beat 'b1'
    duplicate_inherit_beats = [
        {
            "planner_ref": "b1",
            "beat_type": "HOOK",
            "intent": "Hook Part 1",
            "order": 1,
            "target_duration": 15.0,
            "importance": 0.5,
            "evidence_refs": [],
            "inherit_from": "b1",
        },
        {
            "planner_ref": "b2",
            "beat_type": "HOOK",
            "intent": "Hook Part 2",
            "order": 2,
            "target_duration": 15.0,
            "importance": 0.5,
            "evidence_refs": [],
            "inherit_from": "b1",  # Conflict!
        },
    ]
    planner = ContentPlanner(
        repository=repo,
        llm_caller=lambda p: _make_sample_llm_json(beats=duplicate_inherit_beats),
    )

    with pytest.raises(InvalidBeatLineageInheritanceError) as exc_info:
        planner.plan(
            PlannerInput(
                topic="Physics",
                target_video_duration=30.0,
                previous_content_plan_revision_id="rev-uuid-1",
            )
        )
    assert "Duplicate inheritance" in str(exc_info.value)


# 10. Bounded retry works and terminates within max_retries
def test_bounded_retry_behavior():
    # Scenario A: Initial call returns broken JSON, second attempt repairs and succeeds
    call_count = 0
    prompts_received = []

    def healing_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        prompts_received.append(prompt)
        if call_count == 1:
            return "Broken response"
        return _make_sample_llm_json()

    planner = ContentPlanner(llm_caller=healing_llm, max_retries=2)
    plan = planner.plan(
        PlannerInput(
            topic="Quantum",
            target_video_duration=30.0,
            available_evidence_ids=("ev_qubit_1",),
        )
    )

    assert plan is not None
    assert call_count == 2
    assert "[Correction Notice]" in prompts_received[1]

    # Scenario B: Always fails, terminates boundedly after max_retries + 1 calls
    failing_calls = 0

    def always_fail_llm(prompt: str) -> str:
        nonlocal failing_calls
        failing_calls += 1
        return "Still broken"

    planner_failing = ContentPlanner(llm_caller=always_fail_llm, max_retries=2)
    with pytest.raises(PlannerOutputSchemaError):
        planner_failing.plan(PlannerInput(topic="Quantum", target_video_duration=30.0))

    # max_retries=2 means initial attempt (1) + 2 retries = 3 calls total
    assert failing_calls == 3


# 11. Existing Phase 1 plan diff works seamlessly on planner-produced revisions
def test_phase1_plan_diff_works_seamlessly_on_planner_revisions():
    repo = InMemoryContentPlanRepo()

    # Generate initial plan (Revision 1)
    rev1_json = _make_sample_llm_json()
    planner1 = ContentPlanner(repository=repo, llm_caller=lambda p: rev1_json)
    rev1 = planner1.plan(
        PlannerInput(
            topic="Quantum Computing",
            target_video_duration=30.0,
            available_evidence_ids=("ev_qubit_1",),
        )
    )
    assert rev1.revision_number == 1
    assert len(rev1.beats) == 3

    # Generate revision plan (Revision 2):
    # Beat 1 (HOOK) is inherited unchanged
    # Beat 2 (KNOWLEDGE) is inherited with modified intent
    # Beat 3 (CONCLUSION) is removed
    # Beat 4 is a newly added beat
    rev2_beats = [
        {
            "planner_ref": "b1_kept",
            "beat_type": "HOOK",
            "intent": rev1.beats[0].intent,  # Unchanged
            "order": 1,
            "target_duration": 5.0,
            "importance": 0.9,
            "evidence_refs": [],
            "inherit_from": rev1.beats[0].beat_id,
        },
        {
            "planner_ref": "b2_modified",
            "beat_type": "KNOWLEDGE",
            "intent": "Deeper dive into quantum decoherence",  # Modified intent
            "order": 2,
            "target_duration": 15.0,
            "importance": 1.0,
            "evidence_refs": ["ev_qubit_1"],
            "inherit_from": rev1.beats[1].beat_lineage_id,
        },
        {
            "planner_ref": "b4_new",
            "beat_type": "TRANSITION",
            "intent": "Brief transition to closing thoughts",
            "order": 3,
            "target_duration": 10.0,
            "importance": 0.7,
            "evidence_refs": [],
            "inherit_from": None,  # New beat
        },
    ]
    planner2 = ContentPlanner(
        repository=repo,
        llm_caller=lambda p: _make_sample_llm_json(beats=rev2_beats),
    )
    rev2 = planner2.plan(
        PlannerInput(
            topic="Quantum Computing",
            target_video_duration=30.0,
            previous_content_plan_revision_id=rev1.content_plan_revision_id,
            available_evidence_ids=("ev_qubit_1",),
        )
    )
    assert rev2.revision_number == 2

    # Execute Phase 1 diff
    diff = diff_content_plans(rev1, rev2)

    assert diff.old_revision_id == rev1.content_plan_revision_id
    assert diff.new_revision_id == rev2.content_plan_revision_id

    # Verify changes classified properly
    unchanged_changes = [c for c in diff.beat_changes if c.change_type.value == "UNCHANGED"]
    modified_changes = [c for c in diff.beat_changes if c.change_type.value == "MODIFIED"]
    added_changes = [c for c in diff.beat_changes if c.change_type.value == "ADDED"]
    removed_changes = [c for c in diff.beat_changes if c.change_type.value == "REMOVED"]

    assert len(unchanged_changes) == 1
    assert unchanged_changes[0].beat_lineage_id == rev1.beats[0].beat_lineage_id

    assert len(modified_changes) == 1
    assert modified_changes[0].beat_lineage_id == rev1.beats[1].beat_lineage_id

    assert len(added_changes) == 1
    assert added_changes[0].beat_lineage_id == rev2.beats[2].beat_lineage_id

    assert len(removed_changes) == 1
    assert removed_changes[0].beat_lineage_id == rev1.beats[2].beat_lineage_id
