import time
import uuid

from fedn import APIClient

DISCOVER_HOST = '127.0.0.1'
DISCOVER_PORT = 8092
client = APIClient(DISCOVER_HOST, DISCOVER_PORT)

if __name__ == '__main__':

    # Run six sessions, each with 100 rounds.
    num_sessions = 6
    for s in range(num_sessions):

        session_config = {
            "helper": "numpyhelper",
            "id": str(uuid.uuid4()),
            "aggregator": "fedopt",
            "round_timeout": 20,
            "rounds": 100,
            "validate": False,
        }

        session = client.start_session(**session_config)
        if session['success'] is False:
            print(session['message'])
            exit(0)

        print("Started session: {}".format(session))

        # Wait for session to finish
        while not client.session_is_finished(session_config['id']):
            time.sleep(2)
