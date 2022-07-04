import json
import sys
from time import sleep

import fire
import pymongo
import requests

RETRIES = 18
SLEEP = 10


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _retry(try_func, **func_args):
    for _ in range(RETRIES):
        is_success = try_func(**func_args)
        if is_success:
            _eprint('Sucess.')
            return True
        _eprint(f'Sleeping for {SLEEP}.')
        sleep(SLEEP)
    _eprint('Giving up.')
    return False


def _test_rounds(n_rounds):
    client = pymongo.MongoClient(
        "mongodb://fedn_admin:password@localhost:6534")
    collection = client['fedn-test-network']['control']['round']
    query = {'reducer.status': 'Success'}
    n = collection.count_documents(query)
    client.close()
    _eprint(f'Succeded rounds: {n}.')
    return n == n_rounds


def _test_nodes(n_nodes, node_type, reducer_host='localhost', reducer_port='8090'):
    try:
        resp = requests.get(
            f'https://{reducer_host}:{reducer_port}/netgraph', verify=False)
    except Exception as e:
        _eprint(f'Reques exception econuntered: {e}.')
        return False
    if resp.status_code == 200:
        gr = json.loads(resp.content)
        n = sum(values.get('type') == node_type and values.get(
            'status') == 'active' for values in gr['nodes'])
        _eprint(f'Active {node_type}s: {n}.')
        return n == n_nodes
    _eprint(f'Reducer returned {resp.status_code}.')
    return False


def rounds(n_rounds=3):
    _retry(_test_rounds, n_rounds=n_rounds)


def clients(n_clients=2):
    _retry(_test_nodes, n_nodes=n_clients, node_type='client')


def reducer():
    _retry(_test_nodes, n_nodes=1, node_type='reducer')


if __name__ == '__main__':
    fire.Fire()
