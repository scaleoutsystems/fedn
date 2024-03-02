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
import multiprocessing as mp
import time
import uuid
from multiprocessing import Pool, Process

from fedn import APIClient
from fedn.network.clients.client import Client

settings = {
    'DISCOVER_HOST': '127.0.0.1',
    'DISCOVER_PORT': 8092,
    'N_CLIENTS': 1,
    'CLIENTS_ONLINE_FOR_SECONDS': 120
}


client_config = {'discover_host': settings['DISCOVER_HOST'], 'discover_port': settings['DISCOVER_PORT'], 'token': None, 'name': 'testclient',
                 'client_id': 1, 'remote_compute_context': True, 'force_ssl': False, 'dry_run': False, 'secure': False,
                 'preshared_cert': False, 'verify': False, 'preferred_combiner': False,
                 'validator': True, 'trainer': True, 'init': None, 'logfile': 'test.log', 'heartbeat_interval': 2,
                 'reconnect_after_missed_heartbeat': 30}


def run_client(online_for=120, name='client'):
    """ Start a client and stop it after
    online_for seconds.

    """
    conf = copy.deepcopy(client_config)
    conf['name'] = name
    fl_client = Client(conf)
    time.sleep(online_for)
    fl_client.detach()


if __name__ == '__main__':

    processes = []
    for i in range(settings['N_CLIENTS']):
        p = Process(target=run_client, args=(settings['CLIENTS_ONLINE_FOR_SECONDS'], 'client{}'.format(i),))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
