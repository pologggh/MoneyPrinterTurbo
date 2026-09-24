from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.persistence.database_lifecycle import run_database_migrations
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.storyboard_generation_pipeline import (
    StoryboardGenerationInput,
    StoryboardGenerationPipeline,
)


def test_route_plan_executor_rejects_live_session_as_factory() -> None:
    engine = create_engine("sqlite:///:memory:")
    with Session(engine) as session:
        with pytest.raises(TypeError, match="sessionmaker"):
            AssetRoutePlanExecutionService(session_factory=session)


def test_grounded_pipeline_requires_structured_evidence_items() -> None:
    pipeline = StoryboardGenerationPipeline(llm_caller=lambda _: "{}")
    request = StoryboardGenerationInput(
        topic="量子计算",
        source_grounded=True,
        knowledge_context=("这是一段没有稳定证据 ID 的普通正文",),
    )

    with pytest.raises(ValueError, match="evidence_items"):
        pipeline.generate(request)


def test_missing_alembic_config_is_a_startup_error() -> None:
    with (
        patch(
            "app.persistence.database_lifecycle.Path.is_file",
            return_value=False,
        ),
        pytest.raises(FileNotFoundError, match="alembic.ini"),
    ):
        run_database_migrations("sqlite:///:memory:")


def test_redis_queue_knows_agentic_background_workers() -> None:
    from app.controllers.manager.redis_manager import FUNC_MAP

    assert "run_storyboard_assembly_task" in FUNC_MAP
    assert "run_asset_route_plan_task" in FUNC_MAP
