from functools import lru_cache

from src.config import get_settings
from src.services.gemini.client import GeminiClient


@lru_cache(maxsize=1)
def make_gemini_client() -> GeminiClient:
    """
    Create and return a singleton gemini client instance.

    Returns:
        GeminiClient: Configured gemini client
    """
    settings = get_settings()
    return GeminiClient(settings)
