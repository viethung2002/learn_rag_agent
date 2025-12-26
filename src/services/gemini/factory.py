from functools import lru_cache

from src.config import get_settings
from src.services.gemini.client import geminiClient


@lru_cache(maxsize=1)
def make_gemini_client() -> geminiClient:
    """
    Create and return a singleton gemini client instance.

    Returns:
        geminiClient: Configured gemini client
    """
    settings = get_settings()
    return geminiClient(settings)
