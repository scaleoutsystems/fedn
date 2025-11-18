import os
import random  # noqa: F401
from typing import Dict, List, Tuple  # noqa: F401

import numpy as np  # noqa: F401

# TODO: Fix APIClient import
from scaleoututil.api.client import APIClient
from scaleoututil.logging import FednLogger  # noqa: F401
from scaleoututil.serverfunctions.serverfunctionsbase import ServerFunctionsBase  # noqa: F401

if os.getenv("REDUCER_SERVICE_HOST") and os.getenv("REDUCER_SERVICE_PORT"):
    host = f"{os.getenv('REDUCER_SERVICE_HOST')}:{os.getenv('REDUCER_SERVICE_PORT')}/internal"
    port = None
else:
    host = "api-server"
    port = 8092

# TODO: Fix APIClient import
api_client = APIClient(host=host, port=port)

print = FednLogger().info
