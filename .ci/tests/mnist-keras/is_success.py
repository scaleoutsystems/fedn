import pymongo
from time import sleep
import sys

N_ROUNDS = 3
RETRIES= 6
SLEEP=10

def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def _wait_n_rounds(collection):
    n = 0
    for _ in range(RETRIES):
        query = {'reducer.status': 'Success'}
        n = collection.count_documents(query)
        if n == N_ROUNDS:
            return n
        _eprint(f'Succeded rounds {n}. Sleeping for {SLEEP}.')
        sleep(SLEEP)
    _eprint(f'Succeded rounds: {n}. Giving up.')
    return n

if __name__ == '__main__':
    # Connect to mongo
    client = pymongo.MongoClient("mongodb://fedn_admin:password@localhost:6534")

    # Wait for successful rounds
    succeded = _wait_n_rounds(client['fedn-test-network']['control']['round'])
    assert(succeded == N_ROUNDS) # check that all rounds succeeded
    _eprint(f'Succeded rounds: {succeded}. Test passed.')