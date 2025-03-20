from fedn import APIClient
from fedn.cli.shared import get_token, get_project_url

import os
import time

FEDN_NR_ROUNDS = int(os.environ.get("FEDN_NR_ROUNDS", 5))
FEDN_ROUND_TIMEOUT = int(os.environ.get("FEDN_ROUND_TIMEOUT", 180))
FEDN_BUFFER_SIZE = int(os.environ.get("FEDN_BUFFER_SIZE", -1))
FEDN_NR_CLIENTS = int(os.environ.get("FEDN_NR_CLIENTS", 2))
FEDN_CLIENT_TIMEOUT = int(os.environ.get("FEDN_CLIENT_TIMEOUT", 60))
FEDN_FL_ALG = os.environ.get("FEDN_FL_ALG", "fedavg")
FEDN_NR_EXPECTED_AGG = int(os.environ.get("FEDN_NR_EXPECTED_AGG", 2)) # Number of expected aggregated models per combiner
FEDN_SESSION_TIMEOUT = int(os.environ.get("FEDN_SESSION_TIMEOUT", 300)) # Session timeout in seconds, all rounds must be finished within this time
FEDN_SESSION_NAME = os.environ.get("FEDN_SESSION_NAME", "test")


token = get_token(token=None, usr_token=False)
host = get_project_url("", "", None, False)
print(f"Connecting to {host}")
client = APIClient(host=host, token=token, secure=True, verify=False)

start_time = time.time()
while time.time() - start_time < FEDN_CLIENT_TIMEOUT:
    client_obj = client.get_clients()
    if client_obj["count"] == FEDN_NR_CLIENTS and all(c["status"] in ["available", "online"] for c in client_obj["result"]):
        break
    time.sleep(5)  # Wait for 5 seconds before checking again
else:
    raise TimeoutError(f"Not all clients are online within {FEDN_CLIENT_TIMEOUT} seconds")

# Start a new session
result = client.start_session(name=FEDN_SESSION_NAME, 
                     aggregator=FEDN_FL_ALG, 
                     round_timeout=FEDN_ROUND_TIMEOUT, 
                     rounds=FEDN_NR_ROUNDS, 
                     round_buffer_size=FEDN_BUFFER_SIZE,
                     min_clients=FEDN_NR_CLIENTS, 
                     requested_clients=FEDN_NR_CLIENTS)


assert result["message"] == "Session started", f"Expected status 'Session started', got {result['message']}"

session_obj = client.get_sessions()
assert session_obj["count"] == 1, f"Expected 1 session, got {session_obj['count']}"

start_time = time.time()
while time.time() - start_time < FEDN_SESSION_TIMEOUT:
    session_obj = client.get_sessions()
    session_result = session_obj["result"][0]
    if session_result["status"] == "Finished":
        break
    time.sleep(5)  # Wait for 5 seconds before checking again
else:
    raise TimeoutError(f"Session did not finish within {FEDN_SESSION_TIMEOUT} seconds")

assert session_result["status"] == "Finished", "Expected session status 'Finished', got {}".format(session_result["status"])
session_config = session_result["session_config"]
assert session_config["buffer_size"] == FEDN_BUFFER_SIZE, f"Expected buffer size {FEDN_BUFFER_SIZE}, got {session_config['buffer_size']}"
assert session_config["round_timeout"] == FEDN_ROUND_TIMEOUT, f"Expected round timeout {FEDN_ROUND_TIMEOUT}, got {session_config['round_timeout']}"

rounds_obj = client.get_rounds()
assert rounds_obj["count"] == FEDN_NR_ROUNDS, f"Expected {FEDN_NR_ROUNDS} rounds, got {rounds_obj['count']}"
rounds_result = rounds_obj["result"]
for round in rounds_result:
    assert round["status"] == "Finished", f"Expected round status 'Finished', got {round['status']}"
    for combiner in round["combiners"]:
        assert combiner["status"] == "Success", f"Expected combiner status 'Finished', got {combiner['status']}"
        data = combiner["data"]
        assert data["aggregation_time"]["nr_aggregated_models"] == FEDN_NR_EXPECTED_AGG, f"Expected {FEDN_NR_EXPECTED_AGG} aggregated models, got {data['aggregation_time']['nr_aggregated_models']}"
    

validation_obj = client.get_validations()
assert validation_obj["count"] == FEDN_NR_ROUNDS*FEDN_NR_CLIENTS, f"Expected {FEDN_NR_ROUNDS*FEDN_NR_CLIENTS} validations, got {validation_obj['count']}"
# We could assert or test model convergence here

print("All tests passed!", flush=True)



