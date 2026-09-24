from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_editing import (
    CrossBeatMoveNotAllowedError,
    ReorderShotsInput,
    StoryboardEditingService,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    ContentPlanRepository,
    ShotRepository,
    StoryboardRepository,
)
from webui.storyboard_workbench import (
    BEAT_TYPE_LABELS,
    VISUAL_TYPE_LABELS,
    _handle_reorder_shot,
    _handle_save_shot_edit,
    list_available_storyboard_snapshots,
    load_storyboard_view,
)


@pytest.fixture
def test_db_session():
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
def sample_storyboard(test_db_session):
    """Seed DB with a sample multi-beat, multi-shot storyboard."""
    plan_repo = ContentPlanRepository(test_db_session)
    shot_repo = ShotRepository(test_db_session)
    sb_repo = StoryboardRepository(test_db_session)

    b1 = ContentBeat(
        beat_id="beat-inst-1",
        beat_lineage_id="beat-lineage-1",
        order=1,
        intent="Hook the viewer with an intriguing question",
        target_duration=6.0,
        importance=0.9,
        beat_type=BeatType.HOOK,
    )
    b2 = ContentBeat(
        beat_id="beat-inst-2",
        beat_lineage_id="beat-lineage-2",
        order=2,
        intent="Explain self-attention mechanism",
        target_duration=10.0,
        importance=1.0,
        beat_type=BeatType.KNOWLEDGE,
    )
    plan = ContentPlanRevision(
        content_plan_revision_id="plan-rev-1",
        revision_number=1,
        topic="Attention Mechanism Explained",
        overall_target_duration=16.0,
        beats=(b1, b2),
    )
    plan_repo.add_revision(plan)

    s1 = Shot(shot_id="shot-1", beat_lineage_id="beat-lineage-1", local_order=1)
    r1 = ShotRevision(
        shot_revision_id="rev-1",
        shot_id="shot-1",
        revision_number=1,
        beat_lineage_id="beat-lineage-1",
        created_from_beat_instance_id="beat-inst-1",
        narration="Why do transformers need attention?",
        target_duration=6.0,
        visual_goal="Draw attention to neural bottlenecks",
        visual_type=VisualType.DIAGRAM,
        scene_description="Animated bottleneck diagram",
        generation_prompt="schematic diagram showing bottleneck in RNN",
        camera_movement="Static",
    )

    s2 = Shot(shot_id="shot-2", beat_lineage_id="beat-lineage-2", local_order=1)
    r2 = ShotRevision(
        shot_revision_id="rev-2",
        shot_id="shot-2",
        revision_number=1,
        beat_lineage_id="beat-lineage-2",
        created_from_beat_instance_id="beat-inst-2",
        narration="Self-attention computes weights between all tokens.",
        target_duration=5.0,
        visual_goal="Show token interaction matrix",
        visual_type=VisualType.STOCK_VIDEO,
        scene_description="Grid of tokens illuminating simultaneously",
        generation_prompt="cyber grid network nodes lighting up",
        camera_movement="Zoom In",
    )

    s3 = Shot(shot_id="shot-3", beat_lineage_id="beat-lineage-2", local_order=2)
    r3 = ShotRevision(
        shot_revision_id="rev-3",
        shot_id="shot-3",
        revision_number=1,
        beat_lineage_id="beat-lineage-2",
        created_from_beat_instance_id="beat-inst-2",
        narration="This eliminates recurrence completely.",
        target_duration=5.0,
        visual_goal="Highlight speed improvement",
        visual_type=VisualType.AI_VIDEO,
        scene_description="Fast data flow across parallel processors",
        generation_prompt="futuristic parallel data streams",
        camera_movement="Pan Right",
    )

    for shot in (s1, s2, s3):
        shot_repo.add_shot(shot)
    for rev in (r1, r2, r3):
        shot_repo.add_revision(rev)

    snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-test-1",
        content_plan_revision_id="plan-rev-1",
        snapshot_state=StoryboardSnapshotState.DRAFT,
        shot_revision_ids=("rev-1", "rev-2", "rev-3"),
    )
    sb_repo.add_snapshot(snap)

    return {
        "plan": plan,
        "snapshot": snap,
        "shots": (s1, s2, s3),
        "revisions": (r1, r2, r3),
    }


def test_visual_and_beat_type_labels_complete():
    """Verify all domain enum variants are covered by friendly UI mappings."""
    for vt in VisualType:
        assert vt in VISUAL_TYPE_LABELS
        assert vt.value in VISUAL_TYPE_LABELS[vt]

    for bt in BeatType:
        assert bt in BEAT_TYPE_LABELS
        assert bt.value in BEAT_TYPE_LABELS[bt]


def test_list_available_storyboard_snapshots(test_db_session, sample_storyboard, monkeypatch):
    """Verify snapshot discovery queries existing persisted snapshots with topics."""
    @contextmanager
    def mock_get_session():
        yield test_db_session

    monkeypatch.setattr("webui.storyboard_workbench.get_session", mock_get_session)

    snapshots = list_available_storyboard_snapshots()
    assert len(snapshots) == 1
    item = snapshots[0]
    assert item["storyboard_snapshot_id"] == "snap-test-1"
    assert item["topic"] == "Attention Mechanism Explained"
    assert "草稿" in item["label"]
    assert "snap-tes" in item["label"]


def test_load_storyboard_view_deterministic_order(test_db_session, sample_storyboard, monkeypatch):
    """Verify read model groups by Beat.order and Shot.local_order."""
    @contextmanager
    def mock_get_session():
        yield test_db_session

    monkeypatch.setattr("webui.storyboard_workbench.get_session", mock_get_session)

    view = load_storyboard_view("snap-test-1")
    assert view is not None
    assert view.topic == "Attention Mechanism Explained"
    assert view.total_shots == 3
    assert len(view.beats) == 2

    # Beat 1 has 1 shot
    assert view.beats[0].order == 1
    assert len(view.beats[0].shots) == 1
    assert view.beats[0].shots[0].shot_id == "shot-1"
    assert view.beats[0].shots[0].local_order == 1

    # Beat 2 has 2 shots ordered local_order 1, 2
    assert view.beats[1].order == 2
    assert len(view.beats[1].shots) == 2
    assert view.beats[1].shots[0].shot_id == "shot-2"
    assert view.beats[1].shots[0].local_order == 1
    assert view.beats[1].shots[1].shot_id == "shot-3"
    assert view.beats[1].shots[1].local_order == 2


def test_save_shot_edit_flow(test_db_session, sample_storyboard, monkeypatch):
    """Verify editing a shot creates new ShotRevision and new StoryboardSnapshot."""
    @contextmanager
    def mock_get_session():
        yield test_db_session

    monkeypatch.setattr("webui.storyboard_workbench.get_session", mock_get_session)

    view = load_storyboard_view("snap-test-1")
    target_shot = view.beats[0].shots[0]

    session_state_store = {}
    monkeypatch.setattr("streamlit.session_state", session_state_store)
    monkeypatch.setattr("streamlit.toast", lambda msg, icon=None: None)
    monkeypatch.setattr("streamlit.rerun", lambda: None)

    _handle_save_shot_edit(
        view=view,
        shot_id=target_shot.shot_id,
        base_shot_revision_id=target_shot.shot_revision_id,
        narration="Updated hook narration for attention.",
        target_duration=7.5,
        visual_type=VisualType.DIAGRAM,
        visual_goal="Explain attention bottlenecks visually",
        scene_description="High-tech glowing diagram",
        generation_prompt="futuristic glowing node matrix",
        camera_movement="Static",
    )

    new_snap_id = session_state_store["sb_workbench_current_snapshot_id"]
    assert new_snap_id != "snap-test-1"

    # Reload new view to confirm changes
    new_view = load_storyboard_view(new_snap_id)
    assert new_view.storyboard_snapshot_id == new_snap_id
    new_shot = new_view.beats[0].shots[0]
    assert new_shot.shot_id == target_shot.shot_id  # Stable shot identity
    assert new_shot.revision_number == 2            # Incremented revision
    assert new_shot.narration == "Updated hook narration for attention."
    assert new_shot.target_duration == 7.5

    # Original snapshot remains immutable
    orig_view = load_storyboard_view("snap-test-1")
    assert orig_view.beats[0].shots[0].narration == "Why do transformers need attention?"
    assert orig_view.beats[0].shots[0].revision_number == 1


def test_stale_shot_revision_protection(test_db_session, sample_storyboard, monkeypatch):
    """Verify editing with an outdated base_shot_revision_id is safely blocked."""
    @contextmanager
    def mock_get_session():
        yield test_db_session

    monkeypatch.setattr("webui.storyboard_workbench.get_session", mock_get_session)

    view = load_storyboard_view("snap-test-1")
    target_shot = view.beats[0].shots[0]

    session_state_store = {}
    monkeypatch.setattr("streamlit.session_state", session_state_store)
    monkeypatch.setattr("streamlit.toast", lambda msg, icon=None: None)
    monkeypatch.setattr("streamlit.rerun", lambda: None)

    # First edit succeeds and advances to Rev 2
    _handle_save_shot_edit(
        view=view,
        shot_id=target_shot.shot_id,
        base_shot_revision_id=target_shot.shot_revision_id,
        narration="First edit",
        target_duration=6.0,
        visual_type=VisualType.DIAGRAM,
        visual_goal="Goal",
        scene_description="Desc",
        generation_prompt="Prompt",
        camera_movement="Static",
    )
    new_snap_id = session_state_store["sb_workbench_current_snapshot_id"]
    new_view = load_storyboard_view(new_snap_id)

    # Attempting to save another edit against new_view but using the stale Rev 1
    errors_logged = []
    monkeypatch.setattr("streamlit.error", lambda msg: errors_logged.append(msg))

    _handle_save_shot_edit(
        view=new_view,
        shot_id=target_shot.shot_id,
        base_shot_revision_id="rev-1",  # STALE: current in new_view is rev-2!
        narration="Conflicting edit attempt",
        target_duration=6.0,
        visual_type=VisualType.DIAGRAM,
        visual_goal="Goal",
        scene_description="Desc",
        generation_prompt="Prompt",
        camera_movement="Static",
    )

    assert len(errors_logged) == 1
    assert "版本冲突" in errors_logged[0]
    assert "STALE_SHOT_REVISION" in errors_logged[0]


def test_intra_beat_reorder(test_db_session, sample_storyboard, monkeypatch):
    """Verify moving shots up/down inside the same beat."""
    @contextmanager
    def mock_get_session():
        yield test_db_session

    monkeypatch.setattr("webui.storyboard_workbench.get_session", mock_get_session)

    view = load_storyboard_view("snap-test-1")
    beat2 = view.beats[1]
    assert [s.shot_id for s in beat2.shots] == ["shot-2", "shot-3"]

    session_state_store = {}
    monkeypatch.setattr("streamlit.session_state", session_state_store)
    monkeypatch.setattr("streamlit.toast", lambda msg, icon=None: None)
    monkeypatch.setattr("streamlit.rerun", lambda: None)

    # Move shot-3 UP (direction -1)
    _handle_reorder_shot(
        view=view,
        beat_lineage_id=beat2.beat_lineage_id,
        shot_id="shot-3",
        direction=-1,
    )

    new_snap_id = session_state_store["sb_workbench_current_snapshot_id"]
    assert new_snap_id != "snap-test-1"

    new_view = load_storyboard_view(new_snap_id)
    new_beat2 = new_view.beats[1]
    assert [s.shot_id for s in new_beat2.shots] == ["shot-3", "shot-2"]
    assert new_beat2.shots[0].local_order == 1
    assert new_beat2.shots[1].local_order == 2


def test_cross_beat_reorder_disallowed(test_db_session, sample_storyboard):
    """Verify attempting to reorder across beats is strictly rejected."""
    plan_repo = ContentPlanRepository(test_db_session)
    shot_repo = ShotRepository(test_db_session)
    sb_repo = StoryboardRepository(test_db_session)
    service = StoryboardEditingService(plan_repo, shot_repo, sb_repo)

    # Attempting to place shot-1 (from beat 1) into beat 2
    with pytest.raises(CrossBeatMoveNotAllowedError):
        service.reorder_shots_within_beat(
            ReorderShotsInput(
                storyboard_snapshot_id="snap-test-1",
                beat_lineage_id="beat-lineage-2",
                ordered_shot_ids=("shot-1", "shot-2"),
            )
        )
