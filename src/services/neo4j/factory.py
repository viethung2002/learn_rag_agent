"""Factory for Neo4j driver (singleton via lru_cache) and client."""
from functools import lru_cache
from typing import Optional
from neo4j import Driver, GraphDatabase
from src.config import Settings, get_settings
from .client import Neo4jClient


def make_neo4j_driver(settings: Optional[Settings] = None) -> Driver:
    """Return a cached Neo4j driver.
    On app shutdown call ``client.close()`` then ``make_neo4j_driver.cache_clear()``
    so the next process does not reuse a closed driver.
    """
    if settings is None:
        settings = get_settings()
    cfg = settings.neo4j
    return GraphDatabase.driver(
        cfg.uri,
        auth=(cfg.user, cfg.password),
        max_connection_lifetime=cfg.max_connection_lifetime,
        connection_timeout=cfg.connection_timeout,
    )
def make_neo4j_driver_fresh(settings: Optional[Settings] = None) -> Driver:
    """New driver instance (tests or when you avoid the global cache)."""
    if settings is None:
        settings = get_settings()
    cfg = settings.neo4j
    return GraphDatabase.driver(
        cfg.uri,
        auth=(cfg.user, cfg.password),
        max_connection_lifetime=cfg.max_connection_lifetime,
        connection_timeout=cfg.connection_timeout,
    )
def make_neo4j_client(settings: Optional[Settings] = None) -> Neo4jClient:
    """Build a Neo4jClient wrapping the cached driver."""
    if settings is None:
        settings = get_settings()
    return Neo4jClient(make_neo4j_driver(settings), settings)
