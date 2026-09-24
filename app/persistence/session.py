import os
from collections.abc import Callable, Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@localhost:5432/moneyprinterturbo"
)


def get_database_url() -> str:
    """
    Resolves database URL from environment or configuration.

    Order of resolution:
    1. Environment variable DATABASE_URL
    2. config.app.database_url if app config is accessible
    3. Default PostgreSQL local development connection string
    """
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    try:
        from app.config import config

        cfg_url = config.app.get("database_url")
        if cfg_url:
            return cfg_url
    except (ImportError, AttributeError, KeyError):
        return DEFAULT_DATABASE_URL

    return DEFAULT_DATABASE_URL


def create_db_engine(url: str | None = None, **kwargs) -> Engine:
    """Creates a SQLAlchemy engine configured for PostgreSQL/database persistence."""
    resolved_url = url or get_database_url()
    engine_kwargs = {"future": True}
    if resolved_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    engine_kwargs.update(kwargs)
    return create_engine(resolved_url, **engine_kwargs)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Returns a sessionmaker bound to the provided or default engine."""
    eng = engine or create_db_engine()
    return sessionmaker(
        bind=eng, autoflush=False, autocommit=False, expire_on_commit=False
    )


@contextmanager
def get_session(
    session_factory: sessionmaker[Session] | Session | Callable[[], Session] | None = None,
) -> Generator[Session, None, None]:
    """Context manager for acquiring and safely releasing a database session."""
    if isinstance(session_factory, Session):
        try:
            yield session_factory
            session_factory.commit()
        except Exception:
            session_factory.rollback()
            raise
        return

    factory = session_factory or get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
