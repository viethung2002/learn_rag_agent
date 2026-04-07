"""Thin Neo4j client: connectivity check and parameterized Cypher helpers."""

import logging
import time
from typing import Any, Dict, List, Optional

from neo4j import Driver

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Wraps a Neo4j :class:`Driver` with read/write helpers."""

    def __init__(self, driver: Driver, settings: Optional[Settings] = None):
        self._driver = driver
        self._settings = settings or get_settings()
        self._database = self._settings.neo4j.database

    def _redact_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a redacted copy of parameters for safe logging."""
        if not params:
            return {}
        sensitive = {"password", "token", "secret", "api_key"}
        redacted: Dict[str, Any] = {}
        for k, v in params.items():
            if k and k.lower() in sensitive:
                redacted[k] = "<REDACTED>"
            else:
                try:
                    sval = str(v)
                except Exception:
                    sval = "<UNREPRABLE>"
                # cap length to avoid huge logs
                redacted[k] = sval if len(sval) <= 200 else sval[:200] + "..."
        return redacted

    def verify_connectivity(self) -> None:
        """Raise if the database is unreachable."""
        self._driver.verify_connectivity()

    def close(self) -> None:
        self._driver.close()

    def execute_read(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a read query; return rows as dicts (``RETURN`` fields)."""
        params = parameters or {}
        start = time.perf_counter()
        try:
            logger.info("neo4j.query.start", extra={
                "db": self._database,
                "query": query,
                "params": self._redact_params(params),
            })
            with self._driver.session(database=self._database) as session:
                result = session.run(query, params)
                rows = [record.data() for record in result]
            duration = time.perf_counter() - start
            logger.info("neo4j.query.success", extra={
                "db": self._database,
                "rows": len(rows),
                "duration_s": round(duration, 6),
            })
            return rows
        except Exception:
            duration = time.perf_counter() - start
            logger.exception("neo4j.query.error", extra={
                "db": self._database,
                "duration_s": round(duration, 6),
            })
            raise

    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a write query inside a write transaction; return rows if any."""
        params = parameters or {}
        start = time.perf_counter()

        def work(tx):
            result = tx.run(query, params)
            return [record.data() for record in result]

        try:
            logger.info("neo4j.query.start", extra={
                "db": self._database,
                "query": query,
                "params": self._redact_params(params),
            })
            with self._driver.session(database=self._database) as session:
                rows = session.execute_write(work)
            duration = time.perf_counter() - start
            logger.info("neo4j.query.success", extra={
                "db": self._database,
                "rows": len(rows),
                "duration_s": round(duration, 6),
            })
            return rows
        except Exception:
            duration = time.perf_counter() - start
            logger.exception("neo4j.query.error", extra={
                "db": self._database,
                "duration_s": round(duration, 6),
            })
            raise
