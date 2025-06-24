import json
import sys
from time import sleep

import fire
import pymongo
import requests

RETRIES = 30
SLEEP = 20


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _retry(try_func, **func_args):
    for _ in range(RETRIES):
        is_success = try_func(**func_args)
        if is_success:
            _eprint('Success.')
            return True
        _eprint(f'Sleeping for {SLEEP}.')
        sleep(SLEEP)
    _eprint('Giving up.')
    return False


def _test_rounds(n_rounds):
    client = pymongo.MongoClient(
        "mongodb://fedn_admin:password@localhost:6534")
    collection = client['fedn-network']['control']['rounds']
    query = {'status': 'Finished'}
    n = collection.count_documents(query)
    client.close()
    _eprint(f'Succeded rounds: {n}.')
    return n == n_rounds


def _test_nodes(n_nodes, node_type, reducer_host='localhost', reducer_port='8092'):
    try:

        endpoint = "api/v1/clients/" if node_type == "client" else "api/v1/combiners/"

        response = requests.get(
            f'http://{reducer_host}:{reducer_port}/{endpoint}', verify=False)

        if response.status_code == 200:

            data = json.loads(response.content)

            count = 0
            if node_type == "client":
                arr = data.get('result')
                count = sum(element.get('status') == "online" for element in arr)
            else:
                count = data.get('count')

            _eprint(f'Active {node_type}s: {count}.')
            return count == n_nodes

    except Exception as e:
        _eprint(f'Request exception enconuntered: {e}.')
        return False

def _test_controller(reducer_host='localhost', reducer_port='8092'):
    try:
        response = requests.get(
            f'http://{reducer_host}:{reducer_port}/get_controller_status', verify=False)

        if response.status_code == 200:
            data = json.loads(response.content)
            _eprint(f'Controller is running: {data}')
            return True

    except Exception as e:
        _eprint(f'Request exception encountered: {e}.')
        return False


def rounds(n_rounds=3):
    assert (_retry(_test_rounds, n_rounds=n_rounds))


def clients(n_clients=1):
    assert (_retry(_test_nodes, n_nodes=n_clients, node_type='client'))


def combiners(n_combiners=1):
    assert (_retry(_test_nodes, n_nodes=n_combiners, node_type='combiner'))


def reducer():
    assert (_retry(_test_nodes, n_nodes=1, node_type='reducer'))

def controller():
    assert (_retry(_test_controller))


if __name__ == '__main__':
    fire.Fire()
