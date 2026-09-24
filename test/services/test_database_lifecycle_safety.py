from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.persistence.database_lifecycle import (
    _mask_error_message,
    init_database_on_startup,
    mask_database_url,
)


def test_mask_database_url_redacts_credentials():
    """Area 2: Verify database credentials in URLs are masked with ***."""
    url1 = "postgresql://db_user:db_pass_secret123@localhost:5432/moneyprinter"
    masked1 = mask_database_url(url1)
    assert "db_pass_secret123" not in masked1
    assert "db_user" not in masked1
    assert "***:***@localhost:5432/moneyprinter" in masked1

    url2 = "mysql+pymysql://root:P@ssword!@db.internal:3306/testdb"
    masked2 = mask_database_url(url2)
    assert "P@ssword!" not in masked2
    assert "***:***@db.internal:3306/testdb" in masked2

    # SQLite URLs without credentials remain safe and intact
    url3 = "sqlite:///d:/storage/data.db"
    masked3 = mask_database_url(url3)
    assert masked3 == url3


def test_mask_error_message_redacts_passwords():
    """Area 2: Verify error strings containing URLs or passwords are redacted."""
    raw_error = (
        "OperationalError: could not connect to server: "
        "postgresql://admin:super_secret_pw@192.168.1.50:5432/mydb - connection refused"
    )
    masked = _mask_error_message(raw_error)
    assert "super_secret_pw" not in masked
    assert "admin" not in masked
    assert "***:***@192.168.1.50:5432/mydb" in masked


def test_init_database_on_startup_aborts_immediately_on_timeout():
    """
    Area 2: Verify init_database_on_startup aborts without calling migrations
    if wait_for_database fails, eliminating double timeouts.
    """
    with (
        patch("app.persistence.database_lifecycle.wait_for_database", return_value=False) as mock_wait,
        patch("app.persistence.database_lifecycle.run_database_migrations") as mock_migrate,
    ):
        result = init_database_on_startup(timeout=0.1)
        assert result is False
        mock_wait.assert_called_once()
        mock_migrate.assert_not_called()


def test_init_database_on_startup_runs_migrations_on_success():
    """Area 2: Verify migrations run when database connectivity is verified."""
    with (
        patch("app.persistence.database_lifecycle.wait_for_database", return_value=True) as mock_wait,
        patch("app.persistence.database_lifecycle.run_database_migrations") as mock_migrate,
    ):
        result = init_database_on_startup(timeout=1.0)
        assert result is True
        mock_wait.assert_called_once()
        mock_migrate.assert_called_once()


@pytest.mark.asyncio
async def test_asgi_lifespan_raises_on_database_failure():
    """Area 2: Verify ASGI lifespan raises RuntimeError when db initialization fails."""
    from app.asgi import application_lifespan

    app_mock = MagicMock()

    with (
        patch("app.persistence.database_lifecycle.init_database_on_startup", return_value=False),
        pytest.raises(RuntimeError, match="Database initialization"),
    ):
        async with application_lifespan(app_mock):
            pass
