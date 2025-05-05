import os
import random  # noqa: F401
from typing import Dict, List, Tuple  # noqa: F401

import numpy as np  # noqa: F401

from fedn.common.log_config import logger  # noqa: F401
from fedn.network.api.client import APIClient
from fedn.network.combiner.hooks.serverfunctionsbase import ServerFunctionsBase  # noqa: F401

api_client = APIClient(host=os.getenv("REDUCER_SERVICE_HOST", "api-server"), port=int(os.getenv("REDUCER_SERVICE_PORT", 8092)))

print = logger.info

print(api_client.get_active_clients())
