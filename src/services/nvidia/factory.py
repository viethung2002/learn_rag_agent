from functools import lru_cache

from src.config import get_settings
from src.services.nvidia.client import NvidiaClient



@lru_cache(maxsize=1)
def make_nvidia_client() -> NvidiaClient:
    """
    Create and return a singleton nvidia client instance.

    Returns:
        NvidiaClient: Configured nvidia client
    """
    settings = get_settings()
    return NvidiaClient(settings)
