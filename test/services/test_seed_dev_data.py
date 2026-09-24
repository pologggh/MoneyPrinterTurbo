from __future__ import annotations

import pytest

import seed_dev_data


def test_seed_runs_database_migrations_before_opening_session(monkeypatch):
    events: list[str] = []

    def record_migration() -> None:
        events.append("migration")

    class StopAfterInitialization(RuntimeError):
        pass

    def stop_before_seeding():
        raise StopAfterInitialization

    monkeypatch.setattr(seed_dev_data, "run_database_migrations", record_migration)
    monkeypatch.setattr(seed_dev_data, "get_session", stop_before_seeding)

    with pytest.raises(StopAfterInitialization):
        seed_dev_data.seed()

    assert events == ["migration"]
