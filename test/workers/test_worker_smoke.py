from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.stage_executor_registry import get_default_executor_registry
from app.persistence.models import Base
from app.workers.stage_worker import StageWorker, main


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal


def test_stage_worker_lifecycle_smoke(session_factory):
    """Smoke test: StageWorker thread starts, runs loop, and stops gracefully on stop()."""
    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(),
        worker_id="smoke-worker-1",
        poll_interval_seconds=0.05,
    )

    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()

    time.sleep(0.15)
    assert thread.is_alive()
    assert not worker.is_stopped()

    worker.stop()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert worker.is_stopped()


def test_stage_worker_main_cli_smoke(monkeypatch):
    """Smoke test: main() entrypoint parses arguments, connects to db, and responds to stop signal."""
    test_args = [
        "stage_worker.py",
        "--worker-id",
        "cli-test-worker",
        "--poll-interval",
        "0.01",
        "--lease-duration",
        "60",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with (
        patch("app.workers.stage_worker.create_db_engine") as mock_engine,
        patch("app.workers.stage_worker.wait_for_database", return_value=True),
        patch("app.workers.stage_worker.run_database_migrations") as mock_migrate,
        patch.object(StageWorker, "run_forever") as mock_run_forever,
    ):
        main()
        mock_migrate.assert_called_once()
        mock_run_forever.assert_called_once()
