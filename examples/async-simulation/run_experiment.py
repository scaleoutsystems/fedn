import collections
import copy
import json
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np

from fedn import APIClient
from fedn.network.clients.client import Client

DISCOVER_HOST = '127.0.0.1'
DISCOVER_PORT = 8092
client = APIClient(DISCOVER_HOST, DISCOVER_PORT)

if __name__ == '__main__':

    session_config = {
        "helper": "numpyhelper",
        "id": str(uuid.uuid4()),
        "aggregator": "fedavg",
        "round_timeout": 20,
        "rounds": 100,
        "validate": False,
    }

    session = client.start_session(**session_config)
    if session['success'] is False:
        print(session['message'])
        exit(0)

    print("Started session: {}".format(session))

    while not client.session_is_finished(session_config['session_id']):
        time.sleep(2)
