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
    status = client.get_controller_status()
    assert status
    print("Controller status: ", status, flush=True)

    events = client.get_events()
    assert events
    print("Events: ", events, flush=True)

    validations = client.list_validations()
    assert validations
    print("Validations: ", validations, flush=True)

    models = client.get_model_trail()
    assert models
    print("Models: ", models, flush=True)

    clients = client.list_clients()
    assert clients
    print("Clients: ", clients, flush=True)

    combiners = client.list_combiners()
    assert combiners
    print("Combiners: ", combiners, flush=True)

    combiner = client.get_combiner("combiner")
    assert combiner
    print("Combiner: ", combiner, flush=True)

    first_model = client.get_initial_model()
    assert first_model
    print("First model: ", first_model, flush=True)

    package = client.get_package()
    assert package
    print("Package: ", package, flush=True)

    checksum = client.get_package_checksum()
    assert checksum
    print("Checksum: ", checksum, flush=True)

    rounds = client.list_rounds()
    assert rounds
    print("Rounds: ", rounds, flush=True)

    round = client.get_round(1)
    assert round
    print("Round: ", round, flush=True)

    sessions = client.list_sessions()
    assert sessions
    print("Sessions: ", sessions, flush=True)


if __name__ == '__main__':

    client = APIClient(host="localhost", port=8092)
    fire.Fire({
        'set_seed': client.set_initial_model,
        'set_package': client.set_package,
        'start_session': client.start_session,
        'get_client_config': _download_config,
        'test_api_get_methods': test_api_get_methods,
    })
