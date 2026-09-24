from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import config
from app.controllers.v1.storyboard import router
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.persistence.models import Base
from app.persistence.repositories import (
    ContentPlanRepository,
    ShotRepository,
    StoryboardRepository,
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
def client(test_db_session, monkeypatch):
    original_config = dict(config.app)
    config.app["api_key"] = ""

    @contextmanager
    def mock_get_session():
        yield test_db_session

    monkeypatch.setattr("app.controllers.v1.storyboard.get_session", mock_get_session)

    app = FastAPI()
    app.include_router(router)
    test_client = TestClient(app)

    yield test_client

    config.app.clear()
    config.app.update(original_config)


def test_storyboard_api_not_found(client):
    """Verify 404 responses for non-existent entities."""
    # 1. Non-existent storyboard
    res_get = client.get("/api/v1/storyboards/non-existent-snap-id")
    assert res_get.status_code == 404

    # 2. Non-existent shot edit
    res_edit = client.post(
        "/api/v1/storyboards/non-existent-snap-id/shots/shot-1",
        json={"base_shot_revision_id": "rev-1", "narration": "New text"},
    )
    assert res_edit.status_code == 404

    # 3. Non-existent reorder
    res_reorder = client.post(
        "/api/v1/storyboards/non-existent-snap-id/beats/lineage-1/reorder",
        json={"ordered_shot_ids": ["shot-1", "shot-2"]},
    )
    assert res_reorder.status_code == 404


def test_storyboard_api_get_and_edit_shot(client, test_db_session):
    """Verify successful GET and POST edit shot via FastAPI."""
    plan_repo = ContentPlanRepository(test_db_session)
    shot_repo = ShotRepository(test_db_session)
    sb_repo = StoryboardRepository(test_db_session)

    beat = ContentBeat(
        beat_id="beat-inst-1",
        beat_lineage_id="beat-1",
        order=1,
        intent="Explain quantum tunneling concept",
        target_duration=4.0,
        importance=0.8,
        beat_type=BeatType.KNOWLEDGE,
    )
    plan = ContentPlanRevision(
        content_plan_revision_id="plan-rev-1",
        revision_number=1,
        topic="Quantum Tunneling",
        overall_target_duration=4.0,
        beats=(beat,),
    )
    plan_repo.add_revision(plan)

    shot = Shot(shot_id="shot-1", beat_lineage_id="beat-1", local_order=1)
    rev = ShotRevision(
        shot_revision_id="rev-1",
        shot_id="shot-1",
        revision_number=1,
        beat_lineage_id="beat-1",
        created_from_beat_instance_id="beat-inst-1",
        narration="Original narration",
        target_duration=4.0,
        visual_goal="Goal 1",
        visual_type=VisualType.DIAGRAM,
        scene_description="Desc 1",
        generation_prompt="Prompt 1",
        camera_movement="Static",
    )
    shot_repo.add_shot(shot)
    shot_repo.add_revision(rev)

    snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-1",
        content_plan_revision_id="plan-rev-1",
        snapshot_state=StoryboardSnapshotState.DRAFT,
        shot_revision_ids=("rev-1",),
    )
    sb_repo.add_snapshot(snap)

    # 1. GET state
    res_get = client.get("/api/v1/storyboards/snap-1")
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["status"] == 200
    body = data["data"]
    assert body["storyboard_snapshot_id"] == "snap-1"
    assert len(body["beats"]) == 1
    assert len(body["beats"][0]["shots"]) == 1
    assert body["beats"][0]["shots"][0]["narration"] == "Original narration"

    # 2. POST edit shot
    res_edit = client.post(
        "/api/v1/storyboards/snap-1/shots/shot-1",
        json={
            "base_shot_revision_id": "rev-1",
            "narration": "Updated narration",
            "target_duration": 5.0,
        },
    )
    assert res_edit.status_code == 200
    res_data = res_edit.json()["data"]
    new_snap_id = res_data["storyboard_snapshot_id"]
    assert new_snap_id != "snap-1"
    new_rev_id = res_data["shot_revision_ids"][0]
    assert new_rev_id != "rev-1"

    # 3. Verify stale edit detection (trying to edit using old rev-1 on new snapshot)
    res_stale = client.post(
        f"/api/v1/storyboards/{new_snap_id}/shots/shot-1",
        json={
            "base_shot_revision_id": "rev-1",
            "narration": "Conflict narration",
        },
    )
    assert res_stale.status_code == 409


def test_storyboard_api_reorder(client, test_db_session):
    """Verify POST reorder shots via FastAPI."""
    plan_repo = ContentPlanRepository(test_db_session)
    shot_repo = ShotRepository(test_db_session)
    sb_repo = StoryboardRepository(test_db_session)

    beat = ContentBeat(
        beat_id="beat-inst-2",
        beat_lineage_id="beat-1",
        order=1,
        intent="Multi-shot beat",
        target_duration=8.0,
        importance=0.9,
        beat_type=BeatType.KNOWLEDGE,
    )
    plan = ContentPlanRevision(
        content_plan_revision_id="plan-rev-2",
        revision_number=1,
        topic="Topic 2",
        overall_target_duration=8.0,
        beats=(beat,),
    )
    plan_repo.add_revision(plan)

    shot1 = Shot(shot_id="shot-1", beat_lineage_id="beat-1", local_order=1)
    rev1 = ShotRevision(
        shot_revision_id="rev-1",
        shot_id="shot-1",
        revision_number=1,
        beat_lineage_id="beat-1",
        created_from_beat_instance_id="beat-inst-2",
        narration="First shot",
        target_duration=4.0,
        visual_goal="Goal 1",
        visual_type=VisualType.STOCK_VIDEO,
        scene_description="Desc 1",
        generation_prompt="Prompt 1",
        camera_movement="Static",
    )
    shot2 = Shot(shot_id="shot-2", beat_lineage_id="beat-1", local_order=2)
    rev2 = ShotRevision(
        shot_revision_id="rev-2",
        shot_id="shot-2",
        revision_number=1,
        beat_lineage_id="beat-1",
        created_from_beat_instance_id="beat-inst-2",
        narration="Second shot",
        target_duration=4.0,
        visual_goal="Goal 2",
        visual_type=VisualType.STOCK_VIDEO,
        scene_description="Desc 2",
        generation_prompt="Prompt 2",
        camera_movement="Static",
    )
    shot_repo.add_shot(shot1)
    shot_repo.add_revision(rev1)
    shot_repo.add_shot(shot2)
    shot_repo.add_revision(rev2)

    snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-2",
        content_plan_revision_id="plan-rev-2",
        snapshot_state=StoryboardSnapshotState.DRAFT,
        shot_revision_ids=("rev-1", "rev-2"),
    )
    sb_repo.add_snapshot(snap)

    # Reorder shots: shot-2 then shot-1
    res_reorder = client.post(
        "/api/v1/storyboards/snap-2/beats/beat-1/reorder",
        json={"ordered_shot_ids": ["shot-2", "shot-1"]},
    )
    assert res_reorder.status_code == 200
    res_data = res_reorder.json()["data"]
    new_snap_id = res_data["storyboard_snapshot_id"]
    assert new_snap_id != "snap-2"
    assert res_data["shot_revision_ids"] == ["rev-2", "rev-1"]

    # Reorder with invalid/missing shot returns 400
    res_invalid = client.post(
        f"/api/v1/storyboards/{new_snap_id}/beats/beat-1/reorder",
        json={"ordered_shot_ids": ["shot-1"]},  # missing shot-2
    )
    assert res_invalid.status_code == 400
