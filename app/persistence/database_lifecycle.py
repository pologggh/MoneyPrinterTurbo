"""
Database lifecycle management for MoneyPrinterTurbo.
Provides readiness polling and programmatic Alembic migration execution
to ensure databases are ready and up-to-date before serving traffic.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.persistence.session import create_db_engine, get_database_url


def mask_database_url(url: str) -> str:
    """Masks credentials in a database URL or error string so username and password are never logged."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        if parts.password or parts.username:
            netloc = f"***:***@{parts.hostname or ''}"
            if parts.port:
                netloc += f":{parts.port}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return url
    except Exception:
        return re.sub(r"://([^:@]+):([^@]+)@", r"://***:***@", str(url))


def _mask_error_message(exc: Exception | str) -> str:
    msg = str(exc)
    return re.sub(r"://([^:@]+):([^@]+)@", r"://***:***@", msg)


def wait_for_database(
    engine: Engine | None = None,
    timeout: float = 30.0,
    retry_interval: float = 1.0,
) -> bool:
    """
    Polls the database engine with 'SELECT 1' until it becomes ready or timeout occurs.

    Args:
        engine: Optional SQLAlchemy engine. If omitted, one is created from get_database_url().
        timeout: Maximum seconds to wait before giving up.
        retry_interval: Seconds between connection attempts.

    Returns:
        True if database connection succeeded, False otherwise.
    """
    eng = engine or create_db_engine()
    deadline = time.monotonic() + max(0.1, timeout)
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        try:
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"Database readiness check passed on attempt {attempt}.")
            return True
        except Exception as exc:  # noqa: BLE001
            masked_err = _mask_error_message(exc)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(
                    f"Database readiness check timed out after {timeout:.1f}s: {masked_err}"
                )
                return False
            sleep_time = min(retry_interval, remaining)
            logger.warning(
                f"Database not ready yet (attempt {attempt}, {masked_err}). Retrying in {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)

    return False


def run_database_migrations(url: str | None = None) -> None:
    """
    Executes all pending Alembic database migrations up to 'head'.

    Args:
        url: Optional database connection URL. If omitted, resolved via get_database_url().
    """
    from alembic import command
    from alembic.config import Config

    db_url = url or get_database_url()
    project_root = Path(__file__).resolve().parent.parent.parent
    alembic_ini_path = project_root / "alembic.ini"

    if not alembic_ini_path.is_file():
        raise FileNotFoundError(f"alembic.ini not found at {alembic_ini_path}")

    logger.info(f"Running database migrations against {mask_database_url(db_url)}...")
    alembic_cfg = Config(str(alembic_ini_path))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    # Ensure script_location is an absolute path so Alembic finds migrations
    alembic_cfg.set_main_option("script_location", str(project_root / "migrations"))

    migration_engine = create_db_engine(db_url)
    connection = migration_engine.connect()
    advisory_lock_key = 1297100630
    lock_acquired = False
    try:
        if connection.dialect.name == "postgresql":
            connection.execute(
                text("SELECT pg_advisory_lock(:lock_key)"),
                {"lock_key": advisory_lock_key},
            )
            connection.commit()
            lock_acquired = True
        alembic_cfg.attributes["connection"] = connection
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully (up to head).")
    except Exception as exc:
        masked_err = _mask_error_message(exc)
        logger.error(f"Failed to apply database migrations: {masked_err}")
        raise
    finally:
        if lock_acquired:
            try:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": advisory_lock_key},
                )
            except Exception as unlock_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to release database migration advisory lock: "
                    f"{_mask_error_message(unlock_exc)}"
                )
        connection.close()
        migration_engine.dispose()


def init_database_on_startup(timeout: float = 30.0) -> bool:
    """
    Entrypoint hook called during API and WebUI startup.
    Waits for DB readiness, then runs all pending Alembic migrations.

    Returns:
        True if DB is ready and migrated, False if DB check failed.
    """
    db_url = get_database_url()
    logger.info(f"Initializing database persistence: {mask_database_url(db_url)}")

    is_ready = wait_for_database(timeout=timeout)
    if not is_ready:
        logger.error("Database connection could not be established; startup aborted.")
        return False

    try:
        run_database_migrations(db_url)
        return True
    except Exception as exc:  # noqa: BLE001
        masked_err = _mask_error_message(exc)
        logger.error(f"Database migration step failed during startup: {masked_err}")
        return False
