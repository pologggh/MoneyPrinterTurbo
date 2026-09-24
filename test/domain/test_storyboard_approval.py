"""Comprehensive tests for Storyboard Approval and Controlled Beat Replanning.

Tests Phase 3.3 requirements:
- Human approval creating NEW immutable APPROVED snapshot.
- Exact ShotRevision freezing (never upgraded to latest).
- Dedicated StoryboardApprovalRecord persistence and traceability.
- Rejection of approving already approved, empty, or invalid snapshots.
- Single-beat replanning generating NEW Shot entities and revision 1 shots.
- Exact reuse of non-target beat shot revisions.
- Guardrail detecting content plan modification requests (CONTENT_REPLAN_REQUIRED).
- Failure isolation leaving base snapshot intact.
- Editing/replanning an APPROVED snapshot creating a new DRAFT snapshot.
- Fast API controller endpoints and Workbench UI handlers.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_agent import (
    StoryboardError,
    StoryboardExecutionResult,
)
from app.domain.storyboard_approval import (
    ApprovalValidationError,
    ContentReplanRequiredError,
    InvalidSnapshotStateForApprovalError,
    StoryboardApprovalService,
    StoryboardBeatReplanError,
    StoryboardBeatReplanInput,
)
from app.domain.storyboard_editing import StoryboardNotFoundError
from app.persistence.models import Base
from app.persistence.repositories import (
    ContentPlanRepository,
    ShotRepository,
    StoryboardApprovalRepository,
    StoryboardRepository,
)
from webui.storyboard_workbench import (
    _get_approval_record_for_snapshot,
    _handle_approve_storyboard,
    _handle_replan_beat,
    load_storyboard_view,
)


@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite session with foreign keys enabled."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    session = session_factory()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def seeded_environment(db_session):
    """Sets up a 2-beat, 3-shot storyboard in DRAFT state."""
    plan_repo = ContentPlanRepository(db_session)
    shot_repo = ShotRepository(db_session)
    sb_repo = StoryboardRepository(db_session)
    appr_repo = StoryboardApprovalRepository(db_session)

    b1 = ContentBeat(
        beat_id="beat-inst-1",
        beat_lineage_id="lineage-beat-1",
        order=1,
        intent="Hook the audience with an engaging question",
        target_duration=6.0,
        importance=0.9,
        beat_type=BeatType.HOOK,
    )
    b2 = ContentBeat(
        beat_id="beat-inst-2",
        beat_lineage_id="lineage-beat-2",
        order=2,
        intent="Explain core knowledge with data",
        target_duration=10.0,
        importance=1.0,
        beat_type=BeatType.KNOWLEDGE,
    )
    plan = ContentPlanRevision(
        content_plan_revision_id="plan-rev-100",
        revision_number=1,
        topic="Neural Network Fundamentals",
        overall_target_duration=16.0,
        beats=(b1, b2),
    )
    plan_repo.add_revision(plan)

    # Shot 1 for Beat 1
    s1 = Shot(shot_id="shot-101", beat_lineage_id="lineage-beat-1", local_order=1)
    r1 = ShotRevision(
        shot_revision_id="rev-101",
        shot_id="shot-101",
        revision_number=1,
        beat_lineage_id="lineage-beat-1",
        created_from_beat_instance_id="beat-inst-1",
        narration="Have you ever wondered how artificial brains learn?",
        target_duration=6.0,
        visual_type=VisualType.STOCK_VIDEO,
        visual_goal="Curiosity hook",
        scene_description="Futuristic digital brain neurons firing with blue light",
        generation_prompt="Futuristic digital neural networks glowing with light",
        camera_movement="Slow zoom in",
    )
    shot_repo.add_shot(s1)
    shot_repo.add_revision(r1)

    # Shot 2 for Beat 2
    s2 = Shot(shot_id="shot-201", beat_lineage_id="lineage-beat-2", local_order=1)
    r2 = ShotRevision(
        shot_revision_id="rev-201",
        shot_id="shot-201",
        revision_number=1,
        beat_lineage_id="lineage-beat-2",
        created_from_beat_instance_id="beat-inst-2",
        narration="Neural networks adjust numeric weights through backpropagation.",
        target_duration=5.0,
        visual_type=VisualType.DIAGRAM,
        visual_goal="Illustrate weight update formula",
        scene_description="Mathematical equations showing delta weight updates",
        generation_prompt="Clean whiteboard diagram of backpropagation equations",
        camera_movement="Static",
    )
    shot_repo.add_shot(s2)
    shot_repo.add_revision(r2)

    # Shot 3 for Beat 2
    s3 = Shot(shot_id="shot-202", beat_lineage_id="lineage-beat-2", local_order=2)
    r3 = ShotRevision(
        shot_revision_id="rev-202",
        shot_id="shot-202",
        revision_number=1,
        beat_lineage_id="lineage-beat-2",
        created_from_beat_instance_id="beat-inst-2",
        narration="Gradient descent minimizes the objective loss function across iterations.",
        target_duration=5.0,
        visual_type=VisualType.AI_VIDEO,
        visual_goal="Loss surface descending animation",
        scene_description="3D landscape contour with ball descending to valley",
        generation_prompt="3D loss surface terrain visualization with optimization path",
        camera_movement="Orbit around point",
    )
    shot_repo.add_shot(s3)
    shot_repo.add_revision(r3)

    # DRAFT Snapshot
    draft_snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-draft-init",
        content_plan_revision_id="plan-rev-100",
        shot_revision_ids=("rev-101", "rev-201", "rev-202"),
        snapshot_state=StoryboardSnapshotState.DRAFT,
    )
    sb_repo.add_snapshot(draft_snap)
    db_session.commit()

    return {
        "session": db_session,
        "plan_repo": plan_repo,
        "shot_repo": shot_repo,
        "sb_repo": sb_repo,
        "appr_repo": appr_repo,
        "plan": plan,
        "draft_snap": draft_snap,
    }


# =============================================================================
# 1. Approval Invariant Tests
# =============================================================================


def test_approve_draft_creates_new_approved_snapshot_with_exact_revisions(
    seeded_environment,
):
    """Approving a DRAFT snapshot produces a NEW snapshot in APPROVED state freezing exact revisions."""
    env = seeded_environment
    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
    )

    record = service.approve_storyboard(
        storyboard_snapshot_id="snap-draft-init",
        approved_by="director_bob",
        user_note="Looks solid for production",
    )

    # 1. New snapshot created
    assert record.approved_storyboard_snapshot_id != "snap-draft-init"
    assert record.source_draft_snapshot_id == "snap-draft-init"

    # 2. Approved snapshot is loaded and verified
    approved_snap = env["sb_repo"].get_snapshot(record.approved_storyboard_snapshot_id)
    assert approved_snap is not None
    assert approved_snap.snapshot_state == StoryboardSnapshotState.APPROVED
    assert approved_snap.content_plan_revision_id == "plan-rev-100"
    assert approved_snap.shot_revision_ids == ("rev-101", "rev-201", "rev-202")

    # 3. Original DRAFT snapshot is untouched (immutability rule)
    original_draft = env["sb_repo"].get_snapshot("snap-draft-init")
    assert original_draft is not None
    assert original_draft.snapshot_state == StoryboardSnapshotState.DRAFT
    assert original_draft.shot_revision_ids == ("rev-101", "rev-201", "rev-202")


def test_approve_persists_storyboard_approval_record(seeded_environment):
    """Approval persists an immutable StoryboardApprovalRecord traceable to draft and approved snapshots."""
    env = seeded_environment
    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
    )

    record = service.approve_storyboard(
        storyboard_snapshot_id="snap-draft-init",
        approved_by="auditor_jane",
        user_note="Quality sign-off completed",
    )

    # Verify audit record persisted in repository
    loaded_record = env["appr_repo"].get_approval_record(record.storyboard_approval_id)
    assert loaded_record is not None
    assert loaded_record.source_draft_snapshot_id == "snap-draft-init"
    assert (
        loaded_record.approved_storyboard_snapshot_id
        == record.approved_storyboard_snapshot_id
    )
    assert loaded_record.content_plan_revision_id == "plan-rev-100"
    assert loaded_record.exact_shot_revision_ids == ("rev-101", "rev-201", "rev-202")
    assert loaded_record.approved_by == "auditor_jane"
    assert loaded_record.user_note == "Quality sign-off completed"

    # Verify lookups by approved snapshot and source draft snapshot
    by_appr = env["appr_repo"].get_approval_by_approved_snapshot_id(
        record.approved_storyboard_snapshot_id
    )
    assert by_appr is not None
    assert by_appr.storyboard_approval_id == record.storyboard_approval_id

    by_src = env["appr_repo"].get_approval_by_source_draft_id("snap-draft-init")
    assert by_src is not None
    assert by_src.storyboard_approval_id == record.storyboard_approval_id


def test_approve_already_approved_snapshot_rejected(seeded_environment):
    """Attempting to approve a snapshot that is already APPROVED raises InvalidSnapshotStateForApprovalError."""
    env = seeded_environment
    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
    )

    # First approval creates an APPROVED snapshot
    rec = service.approve_storyboard("snap-draft-init")

    # Second approval directly on the APPROVED snapshot must be rejected
    with pytest.raises(InvalidSnapshotStateForApprovalError) as exc_info:
        service.approve_storyboard(rec.approved_storyboard_snapshot_id)

    assert "expected DRAFT" in exc_info.value.message or "APPROVED" in exc_info.value.message


def test_approve_validation_empty_shots_rejected(seeded_environment):
    """Snapshot with no shots cannot be approved."""
    env = seeded_environment
    sb_repo = env["sb_repo"]

    empty_snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-empty",
        content_plan_revision_id="plan-rev-100",
        shot_revision_ids=(),
        snapshot_state=StoryboardSnapshotState.DRAFT,
    )
    sb_repo.add_snapshot(empty_snap)

    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
    )

    with pytest.raises(ApprovalValidationError) as exc_info:
        service.approve_storyboard("snap-empty")

    assert "contains zero shots" in exc_info.value.message


def test_approve_validation_empty_narration_rejected(seeded_environment):
    """Snapshot containing a shot with whitespace or empty narration fails approval."""
    env = seeded_environment
    shot_repo = env["shot_repo"]
    sb_repo = env["sb_repo"]

    s_bad = Shot(
        shot_id="shot-bad-narr", beat_lineage_id="lineage-beat-1", local_order=2
    )
    r_bad = ShotRevision(
        shot_revision_id="rev-bad-narr",
        shot_id="shot-bad-narr",
        revision_number=1,
        beat_lineage_id="lineage-beat-1",
        created_from_beat_instance_id="beat-inst-1",
        narration="   ",  # whitespace only
        target_duration=4.0,
        visual_type=VisualType.STOCK_VIDEO,
        visual_goal="Bad narration test",
        scene_description="Empty narration test visual scene",
        generation_prompt="Stock video clip",
        camera_movement="Static",
    )
    shot_repo.add_shot(s_bad)
    shot_repo.add_revision(r_bad)

    bad_snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-bad-narr",
        content_plan_revision_id="plan-rev-100",
        shot_revision_ids=("rev-101", "rev-bad-narr", "rev-201", "rev-202"),
        snapshot_state=StoryboardSnapshotState.DRAFT,
    )
    sb_repo.add_snapshot(bad_snap)

    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
    )

    with pytest.raises(ApprovalValidationError) as exc_info:
        service.approve_storyboard("snap-bad-narr")

    assert "empty narration" in exc_info.value.message


def test_approve_validation_missing_beat_coverage_rejected(seeded_environment):
    """Snapshot omitting shots for any ContentBeat defined in the plan fails approval."""
    env = seeded_environment
    sb_repo = env["sb_repo"]

    # Only include rev-101 (lineage-beat-1), leaving lineage-beat-2 with 0 shots
    partial_snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-partial",
        content_plan_revision_id="plan-rev-100",
        shot_revision_ids=("rev-101",),
        snapshot_state=StoryboardSnapshotState.DRAFT,
    )
    sb_repo.add_snapshot(partial_snap)

    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
    )

    with pytest.raises(ApprovalValidationError) as exc_info:
        service.approve_storyboard("snap-partial")

    assert "has no shots" in exc_info.value.message


def test_approve_nonexistent_snapshot_rejected(seeded_environment):
    """Approving an unknown snapshot ID raises StoryboardNotFoundError."""
    env = seeded_environment
    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
    )

    with pytest.raises(StoryboardNotFoundError):
        service.approve_storyboard("non-existent-id")


# =============================================================================
# 2. Controlled Beat-Level Replan Invariant Tests
# =============================================================================


def test_replan_beat_creates_new_shots_and_new_draft_snapshot(seeded_environment):
    """Replanning a beat creates NEW Shot entities and new DRAFT snapshot while reusing other beats' revisions."""
    env = seeded_environment

    # Mock agent returning two brand new shots for Beat 1
    mock_agent = MagicMock()
    new_s1 = Shot(shot_id="new-shot-1", beat_lineage_id="lineage-beat-1", local_order=1)
    new_r1 = ShotRevision(
        shot_revision_id="new-rev-1",
        shot_id="new-shot-1",
        revision_number=1,
        beat_lineage_id="lineage-beat-1",
        created_from_beat_instance_id="beat-inst-1",
        narration="Reimagined hook narration with high energy.",
        target_duration=3.0,
        visual_type=VisualType.AI_VIDEO,
        visual_goal="Teaser hook dynamic animation",
        scene_description="Fast-paced montage of AI neural connections",
        generation_prompt="High energy montage of futuristic AI visualization",
        camera_movement="Pan right",
    )
    new_s2 = Shot(shot_id="new-shot-2", beat_lineage_id="lineage-beat-1", local_order=2)
    new_r2 = ShotRevision(
        shot_revision_id="new-rev-2",
        shot_id="new-shot-2",
        revision_number=1,
        beat_lineage_id="lineage-beat-1",
        created_from_beat_instance_id="beat-inst-1",
        narration="Supporting teaser graphic.",
        target_duration=3.0,
        visual_type=VisualType.DIAGRAM,
        visual_goal="Teaser schematic diagram",
        scene_description="Technical animated diagram schematic",
        generation_prompt="Vector diagram explaining model layers",
        camera_movement="Static",
    )
    mock_agent.plan_beat.return_value = StoryboardExecutionResult(
        shots=(new_s1, new_s2),
        shot_revisions=(new_r1, new_r2),
    )

    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
        agent=mock_agent,
    )

    replan_input = StoryboardBeatReplanInput(
        base_storyboard_snapshot_id="snap-draft-init",
        target_beat_lineage_id="lineage-beat-1",
        user_instruction="Make the hook more visually dynamic",
    )
    result = service.replan_beat(replan_input)

    # 1. Verify result summary
    assert result.previous_storyboard_snapshot_id == "snap-draft-init"
    assert result.new_storyboard_snapshot_id != "snap-draft-init"
    assert result.beat_lineage_id == "lineage-beat-1"
    assert result.old_shot_revision_ids == ("rev-101",)
    assert result.new_shot_revision_ids == ("new-rev-1", "new-rev-2")
    assert result.generated_shot_count == 2
    assert result.state == StoryboardSnapshotState.DRAFT

    # 2. Verify new snapshot in DB
    new_snap = env["sb_repo"].get_snapshot(result.new_storyboard_snapshot_id)
    assert new_snap is not None
    assert new_snap.snapshot_state == StoryboardSnapshotState.DRAFT
    # Revisions sequence: (new_rev_1, new_rev_2, rev-201, rev-202)
    assert new_snap.shot_revision_ids == (
        "new-rev-1",
        "new-rev-2",
        "rev-201",
        "rev-202",
    )

    # 3. Verify target beat old shot 'shot-101' is NOT in new snapshot
    new_snap_revisions = env["sb_repo"].get_snapshot_shot_revisions(
        result.new_storyboard_snapshot_id
    )
    new_snap_shot_ids = {r.shot_id for r in new_snap_revisions}
    assert "shot-101" not in new_snap_shot_ids
    assert "new-shot-1" in new_snap_shot_ids
    assert "new-shot-2" in new_snap_shot_ids

    # 4. Verify non-target beat revisions ('rev-201', 'rev-202') were strictly reused
    assert "rev-201" in new_snap.shot_revision_ids
    assert "rev-202" in new_snap.shot_revision_ids

    # 5. Verify base snapshot is untouched in DB
    base_snap = env["sb_repo"].get_snapshot("snap-draft-init")
    assert base_snap.shot_revision_ids == ("rev-101", "rev-201", "rev-202")
    assert base_snap.snapshot_state == StoryboardSnapshotState.DRAFT


def test_replan_beat_content_replan_required_guardrail(seeded_environment):
    """User instructions attempting to alter global content plan trigger ContentReplanRequiredError."""
    env = seeded_environment
    dummy_agent = MagicMock()
    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
        agent=dummy_agent,
    )

    forbidden_prompts = [
        "请删除第二段，换成别的内容",
        "增加新段落讲注意力机制",
        "把整个视频主题换成量子力学",
        "重新规划整体大纲结构",
        "delete this beat from the plan",
        "add a new beat for conclusion",
        "change the video topic completely",
    ]

    for instruction in forbidden_prompts:
        replan_input = StoryboardBeatReplanInput(
            base_storyboard_snapshot_id="snap-draft-init",
            target_beat_lineage_id="lineage-beat-1",
            user_instruction=instruction,
        )
        with pytest.raises(ContentReplanRequiredError) as exc_info:
            service.replan_beat(replan_input)
        assert exc_info.value.code == "CONTENT_REPLAN_REQUIRED"

    # Base snapshot remains untouched
    base_snap = env["sb_repo"].get_snapshot("snap-draft-init")
    assert base_snap.shot_revision_ids == ("rev-101", "rev-201", "rev-202")


def test_replan_beat_agent_failure_leaves_base_snapshot_untouched(seeded_environment):
    """When StoryboardAgent encounters an unrecoverable error, base snapshot remains untouched."""
    env = seeded_environment
    failing_agent = MagicMock()
    failing_agent.plan_beat.side_effect = StoryboardError("LLM quota exhausted")

    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
        agent=failing_agent,
    )

    replan_input = StoryboardBeatReplanInput(
        base_storyboard_snapshot_id="snap-draft-init",
        target_beat_lineage_id="lineage-beat-1",
        user_instruction="Make it concise",
    )

    with pytest.raises(StoryboardBeatReplanError) as exc_info:
        service.replan_beat(replan_input)

    assert "STORYBOARD_REPLAN_FAILED" in exc_info.value.message

    # Base snapshot unchanged
    base_snap = env["sb_repo"].get_snapshot("snap-draft-init")
    assert base_snap.shot_revision_ids == ("rev-101", "rev-201", "rev-202")


def test_editing_or_replanning_approved_snapshot_creates_new_draft(seeded_environment):
    """Replanning or editing from an APPROVED snapshot creates a NEW DRAFT snapshot without mutating the approved one."""
    env = seeded_environment
    service = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
    )

    # 1. Approve initial draft -> snap-appr-1
    rec = service.approve_storyboard("snap-draft-init", approved_by="senior_editor")
    appr_snap_id = rec.approved_storyboard_snapshot_id

    # 2. Replan Beat 2 from the APPROVED snapshot
    mock_agent = MagicMock()
    s_new = Shot(shot_id="shot-replanned", beat_lineage_id="lineage-beat-2", local_order=1)
    r_new = ShotRevision(
        shot_revision_id="rev-replanned",
        shot_id="shot-replanned",
        revision_number=1,
        beat_lineage_id="lineage-beat-2",
        created_from_beat_instance_id="beat-inst-2",
        narration="Single consolidated summary shot for beat 2.",
        target_duration=10.0,
        visual_type=VisualType.STOCK_VIDEO,
        visual_goal="Consolidated summary goal",
        scene_description="Wide shot of summary graphics",
        generation_prompt="Infographic summarizing AI neural networks",
        camera_movement="Static",
    )
    mock_agent.plan_beat.return_value = StoryboardExecutionResult(
        shots=(s_new,),
        shot_revisions=(r_new,),
    )
    service_with_mock = StoryboardApprovalService(
        plan_repository=env["plan_repo"],
        shot_repository=env["shot_repo"],
        storyboard_repository=env["sb_repo"],
        approval_repository=env["appr_repo"],
        agent=mock_agent,
    )

    replan_res = service_with_mock.replan_beat(
        StoryboardBeatReplanInput(
            base_storyboard_snapshot_id=appr_snap_id,
            target_beat_lineage_id="lineage-beat-2",
            user_instruction="Condense into one shot",
        )
    )

    # 3. New snapshot must be DRAFT
    new_snap = env["sb_repo"].get_snapshot(replan_res.new_storyboard_snapshot_id)
    assert new_snap.snapshot_state == StoryboardSnapshotState.DRAFT
    assert replan_res.state == StoryboardSnapshotState.DRAFT

    # 4. The APPROVED snapshot must remain untouched in APPROVED state
    frozen_appr = env["sb_repo"].get_snapshot(appr_snap_id)
    assert frozen_appr.snapshot_state == StoryboardSnapshotState.APPROVED
    assert frozen_appr.shot_revision_ids == ("rev-101", "rev-201", "rev-202")


# =============================================================================
# 3. Fast API Controller Tests
# =============================================================================


def test_api_approve_and_replan_endpoints(seeded_environment, monkeypatch):
    """Verifies the REST API endpoints for approve, replan, and approval metadata."""
    from fastapi import FastAPI

    from app.controllers import base
    from app.controllers.v1.storyboard import router as storyboard_router

    env = seeded_environment

    @contextmanager
    def mock_get_session():
        yield env["session"]

    monkeypatch.setattr("app.controllers.v1.storyboard.get_session", mock_get_session)
    test_app = FastAPI()
    test_app.include_router(storyboard_router)
    test_app.dependency_overrides[base.verify_token] = lambda: True

    client = TestClient(test_app)

    try:
        # 1. POST /storyboards/{id}/approve
        resp_appr = client.post(
            "/api/v1/storyboards/snap-draft-init/approve",
            json={"approved_by": "api_tester", "user_note": "Approved via API"},
        )
        assert resp_appr.status_code == 200
        data_appr = resp_appr.json()["data"]
        approved_id = data_appr["approved_storyboard_snapshot_id"]
        assert data_appr["source_draft_snapshot_id"] == "snap-draft-init"
        assert data_appr["approved_by"] == "api_tester"

        # 2. GET /storyboards/{id}/approval
        resp_get_appr = client.get(f"/api/v1/storyboards/{approved_id}/approval")
        assert resp_get_appr.status_code == 200
        data_rec = resp_get_appr.json()["data"]
        assert data_rec["approved_storyboard_snapshot_id"] == approved_id
        assert data_rec["source_draft_snapshot_id"] == "snap-draft-init"

        # 3. POST /storyboards/{id}/beats/{beat_lineage_id}/replan guardrail error
        resp_replan_fail = client.post(
            "/api/v1/storyboards/snap-draft-init/beats/lineage-beat-1/replan",
            json={"user_instruction": "请删除该段大纲并重构"},
        )
        assert resp_replan_fail.status_code == 400
        assert "CONTENT_REPLAN_REQUIRED" in resp_replan_fail.json()["detail"]

    finally:
        test_app.dependency_overrides.clear()


# =============================================================================
# 4. Workbench UI Handler Tests
# =============================================================================


def test_workbench_ui_handlers(seeded_environment, monkeypatch):
    """Verifies the UI helper functions _handle_approve_storyboard and _handle_replan_beat."""
    import streamlit as st

    env = seeded_environment

    @contextmanager
    def mock_get_session():
        yield env["session"]

    monkeypatch.setattr("webui.storyboard_workbench.get_session", mock_get_session)
    monkeypatch.setattr(st, "rerun", lambda: None)
    monkeypatch.setattr(st, "toast", lambda *a, **kw: None)

    st.session_state["sb_workbench_current_snapshot_id"] = "snap-draft-init"
    st.session_state["sb_workbench_selected_shot_id"] = "shot-101"

    view = load_storyboard_view("snap-draft-init")
    assert view is not None

    # Test _handle_approve_storyboard
    appr_rec = _handle_approve_storyboard(
        view=view,
        approved_by="ui_tester",
        user_note="Approved from UI modal",
    )
    assert appr_rec is not None
    assert (
        st.session_state["sb_workbench_current_snapshot_id"]
        == appr_rec.approved_storyboard_snapshot_id
    )

    # Test audit record fetch helper
    audit_rec = _get_approval_record_for_snapshot(
        appr_rec.approved_storyboard_snapshot_id
    )
    assert audit_rec is not None
    assert audit_rec.approved_by == "ui_tester"

    # Test _handle_replan_beat with mock agent
    mock_agent = MagicMock()
    mock_agent.plan_beat.return_value = StoryboardExecutionResult(
        shots=(
            Shot(shot_id="ui-shot-1", beat_lineage_id="lineage-beat-1", local_order=1),
        ),
        shot_revisions=(
            ShotRevision(
                shot_revision_id="ui-rev-1",
                shot_id="ui-shot-1",
                revision_number=1,
                beat_lineage_id="lineage-beat-1",
                created_from_beat_instance_id="beat-inst-1",
                narration="New narration from UI replan trigger.",
                target_duration=6.0,
                visual_type=VisualType.STOCK_VIDEO,
                visual_goal="UI hook goal",
                scene_description="UI test scene description",
                generation_prompt="UI test prompt",
                camera_movement="Static",
            ),
        ),
    )

    replan_res = _handle_replan_beat(
        view=view,
        beat_lineage_id="lineage-beat-1",
        user_instruction="Keep it vivid",
        agent=mock_agent,
    )
    assert replan_res is not None
    assert (
        st.session_state["sb_workbench_current_snapshot_id"]
        == replan_res.new_storyboard_snapshot_id
    )
    assert replan_res.generated_shot_count == 1
