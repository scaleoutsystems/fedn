"""This scripts starts N_CLIENTS using the SDK.

If you are running with a local deploy of FEDn
using docker compose, you need to make sure that clients
are able to resolve the name "combiner" to 127.0.0.1

One way to accomplish this is to edit your /etc/host,
adding the line:

combiner    127.0.0.1

"""


import copy
import time

from fedn import APIClient
from fedn.network.clients.client import Client

DISCOVER_HOST = '127.0.0.1'
DISCOVER_PORT = 8092
N_CLIENTS = 5
CLIENTS_AVAILABLE_TIME = 120

config = {'discover_host': DISCOVER_HOST, 'discover_port': DISCOVER_PORT, 'token': None, 'name': 'testclient',
          'client_id': 1, 'remote_compute_context': True, 'force_ssl': False, 'dry_run': False, 'secure': False,
          'preshared_cert': False, 'verify': False, 'preferred_combiner': False,
          'validator': True, 'trainer': True, 'init': None, 'logfile': 'test.log', 'heartbeat_interval': 2,
          'reconnect_after_missed_heartbeat': 30}

if __name__ == '__main__':

    # Start up N_CLIENTS clients
    clients = []
    for i in range(N_CLIENTS):
        config_i = copy.deepcopy(config)
        config['name'] = 'client{}'.format(i)
        clients.append(Client(config))

    # Run a session
    client = APIClient(DISCOVER_HOST, DISCOVER_PORT)

    session_config_fedavg = {
        "helper": "numpyhelper",
        "session_id": str(uuid.uuid4()),
        "aggregator": "fedavg",
        "round_timeout": 30,
        "rounds": 5,
    }

    result_fedavg = client.start_session(**session_config_fedavg)
    while not client.session_is_finished(session_config_fedavg['session_id']):
        time.sleep(1)

    # Disconnect clients
    time.sleep(CLIENTS_AVAILABLE_TIME)
    for client in clients:
        client.detach()
