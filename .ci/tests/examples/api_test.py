import fire
import yaml
from server_functions import ServerFunctions

from fedn import APIClient


def _download_config(output):
    """ Download the client configuration file from the controller.

    :param output: The output file path.
    :type output: str
    """
    client = APIClient(host="localhost", port=8092)
    config = client.get_client_config(checksum=True)
    with open(output, 'w') as f:
        f.write(yaml.dump(config))


def test_api_get_methods():
    client = APIClient(host="localhost", port=8092)

    # --- Clients --- #
    clients = client.get_clients()
    assert clients
    print("Clients: ", clients, flush=True)

    active_clients = client.get_active_clients()
    assert active_clients
    print("Active clients: ", active_clients, flush=True)

    clients_count = client.get_clients_count()
    assert clients_count
    print("Clients count: ", clients_count, flush=True)

    client_id = clients["result"][0]["client_id"]
    client_obj = client.get_client(client_id)
    assert client_obj
    assert client_id == client_obj["client_id"]
    print("Client: ", client_obj, flush=True)
    
    assert clients_count == len(clients["result"])
    assert clients_count == clients["count"]

    # --- Combiners --- #

    combiners = client.get_combiners()
    assert combiners
    print("Combiners: ", combiners, flush=True)

    combiners_count = client.get_combiners_count()
    assert combiners_count
    print("Combiners count: ", combiners_count, flush=True)

    combiner = client.get_combiner("combiner")
    assert combiner
    print("Combiner: ", combiner, flush=True)

    assert combiners_count == len(combiners["result"])
    assert combiners_count == combiners["count"]

    # --- Controllers --- #

    status = client.get_controller_status()
    assert status
    print("Controller status: ", status, flush=True)

    # --- Models --- #

    models = client.get_models()
    assert models
    print("Models: ", models, flush=True)

    models_count = client.get_models_count()
    assert models_count
    print("Models count: ", models_count, flush=True)

    models_from_trail = client.get_model_trail()
    assert models_from_trail
    assert len(models_from_trail) == models_count
    print("Models (model trail): ", models_from_trail, flush=True)

    active_model = client.get_active_model()
    assert active_model
    print("Active model: ", active_model, flush=True)

    assert models_count == len(models["result"])
    assert models_count == models["count"]
    
    # --- Packages --- #

    packages = client.get_packages()
    assert packages
    print("Packages: ", packages, flush=True)

    packages_count = client.get_packages_count()
    assert packages_count
    print("Packages count: ", packages_count, flush=True)

    active_package = client.get_active_package()
    assert active_package
    print("Active package: ", active_package, flush=True)

    checksum = client.get_package_checksum()
    assert checksum
    print("Checksum: ", checksum, flush=True)

    assert packages_count == len(packages["result"])
    assert packages_count == packages["count"]

    # --- Rounds --- #

    rounds = client.get_rounds()
    assert rounds
    print("Rounds: ", rounds, flush=True)

    rounds_count = client.get_rounds_count()
    assert rounds_count
    print("Rounds count: ", rounds_count, flush=True)

    assert rounds_count == len(rounds["result"])
    assert rounds_count == rounds["count"]

    # --- Sessions --- #

    sessions = client.get_sessions()
    assert sessions
    print("Sessions: ", sessions, flush=True)

    sessions_count = client.get_sessions_count()
    assert sessions_count
    print("Sessions count: ", sessions_count, flush=True)

    session = client.get_session(id=sessions["result"][0]["session_id"])
    assert session
    assert session["session_id"] == sessions["result"][0]["session_id"]
    print("Session: ", session, flush=True)

    assert sessions_count == len(sessions["result"])
    assert sessions_count == sessions["count"]

    # --- Statuses --- #

    statuses = client.get_statuses()
    assert statuses
    print("Statuses: ", statuses, flush=True)

    statuses_count = client.get_statuses_count()
    assert statuses_count
    print("Statuses count: ", statuses_count, flush=True)

    assert statuses_count == len(statuses["result"])
    assert statuses_count == statuses["count"]

    # --- Validations --- #

    validations = client.get_validations()
    assert validations
    print("Validations: ", validations, flush=True)

    validations_count = client.get_validations_count()
    assert validations_count
    print("Validations count: ", validations_count, flush=True)

    assert validations_count == len(validations["result"])
    assert validations_count == validations["count"]


def start_sf_session(name, rounds, helper):
    client = APIClient(host="localhost", port=8092)
    client.start_session(name=name, rounds=rounds, helper=helper, server_functions=ServerFunctions)

if __name__ == '__main__':

    client = APIClient(host="localhost", port=8092)
    fire.Fire({
        'set_seed': client.set_active_model,
        'set_package': client.set_active_package,
        'start_session': client.start_session,
        'start_sf_session': start_sf_session,
        'get_client_config': _download_config,
        'test_api_get_methods': test_api_get_methods,
    })
