"""This scripts starts N_CLIENTS using the SDK.





If you are running with a local deploy of FEDn
using docker compose, you need to make sure that clients
are able to resolve the name "combiner" to 127.0.0.1

One way to accomplish this is to edit your /etc/host,
adding the line:

combiner    127.0.0.1

(this requires root previliges)
"""

import copy
import time
from multiprocessing import Process

import numpy as np

from fedn.network.clients.client import Client

settings = {
    'DISCOVER_HOST': '127.0.0.1',
    'DISCOVER_PORT': 8092,
    'N_CLIENTS': 10,
    'N_CYCLES': 100,
    'CLIENTS_MAX_DELAY': 10,
    'CLIENTS_ONLINE_FOR_SECONDS': 120
}

client_config = {'discover_host': settings['DISCOVER_HOST'], 'discover_port': settings['DISCOVER_PORT'], 'token': None, 'name': 'testclient',
                 'client_id': 1, 'remote_compute_context': True, 'force_ssl': False, 'dry_run': False, 'secure': False,
                 'preshared_cert': False, 'verify': False, 'preferred_combiner': False,
                 'validator': True, 'trainer': True, 'init': None, 'logfile': 'test.log', 'heartbeat_interval': 2,
                 'reconnect_after_missed_heartbeat': 30}


def run_client(online_for=120, name='client'):
    """ Simulates a client that starts and stops
    at random intervals.

    The client will start after a radom time 'mean_delay',
    stay online for 'online_for' seconds (deterministic),
    then disconnect.

    This is repeated for N_CYCLES.

    """

    conf = copy.deepcopy(client_config)
    conf['name'] = name

    for i in range(settings['N_CYCLES']):
        # Sample a delay until the client starts
        t_start = np.random.randint(0, settings['CLIENTS_MAX_DELAY'])
        time.sleep(t_start)
        fl_client = Client(conf)
        time.sleep(online_for)
        fl_client.disconnect()


if __name__ == '__main__':

    # We start N_CLIENTS independent client processes
    processes = []
    for i in range(settings['N_CLIENTS']):
        p = Process(target=run_client, args=(settings['CLIENTS_ONLINE_FOR_SECONDS'], 'client{}'.format(i),))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
