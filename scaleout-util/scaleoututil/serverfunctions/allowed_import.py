import os
import random  # noqa: F401
from typing import Optional, Dict, List, Tuple  # noqa: F401

import numpy as np  # noqa: F401

from scaleoututil.api.client import APIClient
from scaleoututil.logging import ScaleoutLogger  # noqa: F401
from scaleoututil.serverfunctions.serverfunctionsbase import ServerFunctionsBase, RoundType  # noqa: F401


_api_client_instance: Optional[APIClient] = None


def _get_api_client() -> APIClient:
    """Get or create the API client instance (lazy initialization).

    This function ensures the API client is only created when first accessed,
    avoiding initialization errors during module import.
    """
    global _api_client_instance
    if _api_client_instance is None:
        if os.getenv("REDUCER_SERVICE_HOST") and os.getenv("REDUCER_SERVICE_PORT"):
            host = f"{os.getenv('REDUCER_SERVICE_HOST')}:{os.getenv('REDUCER_SERVICE_PORT')}/internal"
            port = None
        else:
            host = "scaleout-api-server"
            port = 8092
        _api_client_instance = APIClient(host=host, port=port)
    return _api_client_instance


# Provide api_client as a module-level attribute with lazy initialization
class _APIClientProxy:
    """Proxy object that lazily initializes the API client on first attribute access."""

    def __getattr__(self, name):
        return getattr(_get_api_client(), name)

    def __call__(self, *args, **kwargs):
        return _get_api_client()(*args, **kwargs)


api_client = _APIClientProxy()
print = ScaleoutLogger().info


# --- Combiner context ---
_COMBINER_NAME: Optional[str] = None


# combiner id can be useful e.g. for sharing attributes across sessions and combiners using the api client
def get_combiner_name() -> str:
    """Return the ID of the current combiner.

    The combiner ID is injected by the Scaleout runtime and is only
    available while code is executing in a combiner context. It can be
    used, for example, together with :data:`api_client` to share
    attributes or state across sessions and combiners.

    Returns:
        str: The identifier of the current combiner.

    Raises:
        RuntimeError: If the combiner ID has not been set yet, for
            example when called outside of a combiner context or before
            the runtime has initialised it.

    """
    if _COMBINER_NAME is None:
        raise RuntimeError("combiner_name not set.")
    return _COMBINER_NAME
