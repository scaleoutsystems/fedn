import copy
import time

from fedn.network.clients.client import Client

DISCOVER_HOST = '34.207.219.112'
DISCOVER_PORT = 8092
N_CLIENTS = 10

config = {'discover_host': DISCOVER_HOST, 'discover_port': DISCOVER_PORT, 'token': None, 'name': 'testclient',
          'client_id': 1, 'remote_compute_context': True, 'force_ssl': False, 'dry_run': False, 'secure': False,
          'preshared_cert': False, 'verify': False, 'preferred_combiner': False,
          'validator': False, 'trainer': True, 'init': None, 'logfile': 'test.log', 'heartbeat_interval': 2,
          'reconnect_after_missed_heartbeat': 30}

clients = []
for i in range(N_CLIENTS):
    config_i = copy.deepcopy(config)
    config['name'] = 'client-{}'.format(i)
    clients.append(Client(config))

time.sleep(60)
for client in clients:
    client._detach()
