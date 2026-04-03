"""Thin Neo4j client: connectivity check and parameterized Cypher helpers."""

import logging
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

    def verify_connectivity(self) -> None:
        """Raise if the database is unreachable."""
        self._driver.verify_connectivity()

    def close(self) -> None:
        self._driver.close()

    def execute_read(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a read query; return rows as dicts (``RETURN`` fields)."""
        params = parameters or {}
        with self._driver.session(database=self._database) as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a write query inside a write transaction; return rows if any."""

        params = parameters or {}

        def work(tx):
            result = tx.run(query, params)
            return [record.data() for record in result]

        with self._driver.session(database=self._database) as session:
            return session.execute_write(work)
