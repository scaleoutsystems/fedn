import sys
from time import sleep

import pymongo

N_CLIENTS=2
RETRIES=18
SLEEP=10

def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def _wait_n_rounds(collection):
    n = 0
    for _ in range(RETRIES):
        query = {'type': 'INFERENCE'}
        n = collection.count_documents(query)
        if n == N_CLIENTS:
            return n
        _eprint(f'Succeded rounds {n}. Sleeping for {SLEEP}.')
        sleep(SLEEP)
    _eprint(f'Succeded rounds: {n}. Giving up.')
    return n

if __name__ == '__main__':
    # Connect to mongo
    client = pymongo.MongoClient("mongodb://fedn_admin:password@localhost:6534")

    # Wait for successful rounds
    succeded = _wait_n_rounds(client['fedn-test-network']['control']['status'])
    assert(succeded == N_CLIENTS) # check that all rounds succeeded
    _eprint(f'Succeded inference clients: {succeded}. Test passed.')
