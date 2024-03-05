import collections
import copy
import json
import uuid

import matplotlib.pyplot as plt
import numpy as np

from fedn import APIClient
from fedn.network.clients.client import Client

DISCOVER_HOST = '127.0.0.1'
DISCOVER_PORT = 8092
client = APIClient(DISCOVER_HOST, DISCOVER_PORT)

if __name__ == '__main__':

    session_config_fedavg = {
        "helper": "numpyhelper",
        "session_id": str(uuid.uuid4()),
        "aggregator": "fedavg",
        "round_timeout": 10,
        "rounds": 5,
    }

    result_fedavg = client.start_session(**session_config_fedavg)
