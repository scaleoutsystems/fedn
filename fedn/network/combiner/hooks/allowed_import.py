import os
import random  # noqa: F401
from typing import Dict, List, Tuple  # noqa: F401

import numpy as np  # noqa: F401

from fedn import APIClient
from fedn.common.log_config import logger  # noqa: F401
from fedn.network.combiner.hooks.serverfunctionsbase import ServerFunctionsBase  # noqa: F401

if os.getenv("REDUCER_SERVICE_HOST") and os.getenv("REDUCER_SERVICE_PORT"):
    host = f"{os.getenv('REDUCER_SERVICE_HOST')}:{os.getenv('REDUCER_SERVICE_PORT')}/internal"
    port = None
else:
    host = "api-server"
    port = 8092

api_client = APIClient(host=host, port=port)

print = logger.info
