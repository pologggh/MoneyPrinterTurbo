from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.domain.workflow_state import ArtifactType, Stage


def test_artifact_type_enum_values():
    """Verify supported ArtifactType values are explicit and well-defined."""
    expected_types = {
        "EVIDENCE_SNAPSHOT",
        "CONTENT_PLAN_REVISION",
        "SCRIPT_REVISION",
        "STORYBOARD_SNAPSHOT",
        "ASSET_ROUTE_PLAN",
        "EXECUTION_RUN",
        "AUDIO_OUTPUT",
        "COMPOSITION_OUTPUT",
        "EVALUATION_SNAPSHOT",
        "DELIVERY_MANIFEST",
    }
    actual_types = {t.value for t in ArtifactType}
    assert expected_types.issubset(actual_types)


def test_task_artifact_ref_creation_and_immutability():
    """Verify TaskArtifactRef links task to typed artifact ID without duplicating payload."""
    from app.domain.task_artifact import TaskArtifactRef

    ref_id = uuid4().hex
    task_id = uuid4().hex
    storyboard_id = f"sb_snap_{uuid4().hex[:8]}"
    now = datetime.now(UTC)

    ref = TaskArtifactRef(
        task_artifact_ref_id=ref_id,
        task_id=task_id,
        stage=Stage.STORYBOARD,
        artifact_type=ArtifactType.STORYBOARD_SNAPSHOT,
        artifact_id=storyboard_id,
        artifact_version="v1",
        created_at=now,
        metadata_json={"shot_count": 5},
    )

    assert ref.task_artifact_ref_id == ref_id
    assert ref.task_id == task_id
    assert ref.stage == Stage.STORYBOARD
    assert ref.artifact_type == ArtifactType.STORYBOARD_SNAPSHOT
    assert ref.artifact_id == storyboard_id
    assert ref.artifact_version == "v1"
    assert ref.created_at == now
    assert ref.metadata_json == {"shot_count": 5}

    # Immutability check
    with pytest.raises(Exception):
        ref.artifact_id = "new_id"


def test_task_artifact_ref_helper_constructor():
    """Verify convenience constructor TaskArtifactRef.create()."""
    from app.domain.task_artifact import TaskArtifactRef

    task_id = "task-123"
    ref = TaskArtifactRef.create(
        task_id=task_id,
        stage=Stage.KNOWLEDGE_PLAN,
        artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
        artifact_id="rev-456",
        artifact_version="rev_1",
    )

    assert ref.task_id == task_id
    assert ref.stage == Stage.KNOWLEDGE_PLAN
    assert ref.artifact_type == ArtifactType.CONTENT_PLAN_REVISION
    assert ref.artifact_id == "rev-456"
    assert ref.artifact_version == "rev_1"
    assert len(ref.task_artifact_ref_id) > 0
    assert isinstance(ref.created_at, datetime)
