import json
import sys
from time import sleep

import pandas
import requests

N_CLIENTS = 2
RETRIES = 18
SLEEP = 10
REDUCER_HOST = 'localhost'


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _wait_n_clients():
    n = 0
    for _ in range(RETRIES):
        resp = requests.get(
            f'https://{REDUCER_HOST}:8090/netgraph', verify=False)
        if resp.status_code == 200:
            gr = json.loads(resp.content)
            n = sum(values.get('type') == 'client' and values.get(
                'status') == 'active' for values in gr['nodes'])
            if n == N_CLIENTS:
                return n
        _eprint(f'Connected clients: {n}. Sleeping for {SLEEP}.')
        sleep(SLEEP)
    _eprint(f'Connected clients: {n}. Giving up.')
    return n


if __name__ == '__main__':
    # Wait for clients
    connected = _wait_n_clients()
    assert(connected == N_CLIENTS)  # check that all clients connected
    _eprint(f'Connected clients: {connected}. Test passed.')
