"""Unified factory for OpenSearch client."""

from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings

from .client import OpenSearchClient


@lru_cache(maxsize=1)
def _make_opensearch_client_cached() -> OpenSearchClient:
    """Internal cached creator that does not take Settings as argument to avoid
    passing unhashable Pydantic objects into lru_cache."""
    settings = get_settings()
    return OpenSearchClient(host=settings.opensearch.host, settings=settings)


def make_opensearch_client(settings: Optional[Settings] = None) -> OpenSearchClient:
    """Factory function to create cached OpenSearch client.

    Accepts an optional Settings object for API compatibility but uses an
    internal zero-argument cached creator so the cached key remains hashable.
    """
    # We ignore the passed Settings for caching purposes; use the internal cached instance
    return _make_opensearch_client_cached()


def make_opensearch_client_fresh(settings: Optional[Settings] = None, host: Optional[str] = None) -> OpenSearchClient:
    """Factory function to create a fresh OpenSearch client (not cached).

    Use this when you need a new client instance (e.g., for testing
    or when connection issues occur).

    :param settings: Optional settings instance
    :param host: Optional host override
    :returns: New OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    # Use provided host or settings host
    opensearch_host = host or settings.opensearch.host

    return OpenSearchClient(host=opensearch_host, settings=settings)
