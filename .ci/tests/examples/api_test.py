import fire
import yaml

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
    print("Models: ", models_from_trail, flush=True)

    active_model = client.get_active_model()
    assert active_model
    print("Active model: ", active_model, flush=True)

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

    # --- Rounds --- #

    rounds = client.get_rounds()
    assert rounds
    print("Rounds: ", rounds, flush=True)

    rounds_count = client.get_rounds_count()
    assert rounds_count
    print("Rounds count: ", rounds_count, flush=True)

    # --- Sessions --- #

    sessions = client.get_sessions()
    assert sessions
    print("Sessions: ", sessions, flush=True)

    sessions_count = client.get_sessions_count()
    assert sessions_count
    print("Sessions count: ", sessions_count, flush=True)

    # --- Statuses --- #

    statuses = client.get_statuses()
    assert statuses
    print("Statuses: ", statuses, flush=True)

    statuses_count = client.get_statuses_count()
    assert statuses_count
    print("Statuses count: ", statuses_count, flush=True)

    # --- Validations --- #

    validations = client.get_validations()
    assert validations
    print("Validations: ", validations, flush=True)

    validations_count = client.get_validations_count()
    assert validations_count
    print("Validations count: ", validations_count, flush=True)


def start_sf_session(name, rounds, helper):
    import sys, os # not-floating-import
    sys.path.insert(0, os.path.abspath('/home/runner/work/fedn/fedn/examples/server-functions'))
    from server_functions import ServerFunctions # not-floating-import
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
