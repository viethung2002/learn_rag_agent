"""Helpers for LangGraph Postgres checkpointer connection strings."""

import logging
import re
from urllib.parse import urlparse, urlunparse

from psycopg import AsyncConnection, sql

logger = logging.getLogger(__name__)


def to_psycopg_conninfo(url: str) -> str:
    """Normalize a SQLAlchemy-style URL to a psycopg3 connection string.

    AsyncPostgresSaver uses psycopg3 (``postgresql://``), not ``postgresql+psycopg2://``.
    """
    if url.startswith("postgresql+psycopg2://"):
        return "postgresql://" + url.removeprefix("postgresql+psycopg2://")
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url.removeprefix("postgresql+psycopg://")
    if url.startswith("postgresql://"):
        return url
    raise ValueError(
        "Expected postgresql://, postgresql+psycopg://, or postgresql+psycopg2:// URL"
    )


def mask_conninfo_for_log(conninfo: str) -> str:
    """Hide password in logs."""
    return re.sub(r"(//[^:]+:)[^@]+@", r"\1***@", conninfo)


async def ensure_checkpoint_database_exists(conninfo: str) -> None:
    """Create the checkpoint database if it is missing.

    Docker volumes created before ``langgraph_checkpoint`` was added never run
    ``init-db`` again; this connects to the maintenance DB ``postgres`` and
    creates the target database when needed.
    """
    parsed = urlparse(conninfo)
    path = (parsed.path or "").strip("/")
    if not path or path == "postgres":
        return
    target_db = path
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", target_db):
        raise ValueError(f"Invalid database name in checkpoint URL: {target_db}")

    admin_uri = urlunparse(parsed._replace(path="/postgres"))
    async with await AsyncConnection.connect(
        admin_uri,
        autocommit=True,
        prepare_threshold=0,
    ) as conn:
        cur = await conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (target_db,),
        )
        row = await cur.fetchone()
        if row is not None:
            return

        logger.info(
            "Creating database %s (missing; common when Postgres volume predates init-db)",
            target_db,
        )
        await conn.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db))
        )
