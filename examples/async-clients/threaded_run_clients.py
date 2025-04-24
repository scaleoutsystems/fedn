"""This scripts starts N_CLIENTS using the SDK.


If you are running with a local deploy of FEDn
using docker compose, you need to make sure that clients
are able to resolve the name "combiner" to 127.0.0.1

One way to accomplish this is to edit your /etc/host,
adding the line:

combiner    127.0.0.1

(this requires root previliges)
"""

import threading
import time
import uuid
from io import BytesIO
from multiprocessing import Process

import numpy as np
from init_seed import compile_model, make_data
from sklearn.metrics import accuracy_score

import random
from config import settings
from fedn import FednClient
import json

HELPER_MODULE = "numpyhelper"

log_lock = threading.Lock()

LOG_FILE = "/Users/sigvard/Desktop/client_update.json" # Logging file for all clients


def get_api_url(host: str, port: int = None, secure: bool = False):
    if secure:
        url = f"https://{host}:{port}" if port else f"https://{host}"
    else:
        url = f"http://{host}:{port}" if port else f"http://{host}"
    if not url.endswith("/"):
        url += "/"
    return url


def load_parameters(model_bytes_io: BytesIO):
    """Load model parameters from a BytesIO object."""
    model_bytes_io.seek(0)  # Ensure we're at the start of the BytesIO object
    a = np.load(model_bytes_io)
    weights = [a[str(i)] for i in range(len(a.files))]
    return weights


def load_model(model_bytes_io: BytesIO):
    parameters = load_parameters(model_bytes_io)

    model = compile_model()
    n = len(parameters) // 2
    model.coefs_ = parameters[:n]
    model.intercepts_ = parameters[n:]

    return model

def callback_train(client_id, client_name):
    def on_train(in_model, client_settings):
        print(in_model)
        start_time = time.perf_counter()
        model = load_model(in_model)
        X_train, y_train, _, _ = make_data()
        epochs = settings["N_EPOCHS"]
        for i in range(epochs):
            model.partial_fit(X_train, y_train)

        # Prepare updated model parameters
        updated_parameters = model.coefs_ + model.intercepts_
        out_model = BytesIO()
        np.savez_compressed(out_model, **{str(i): w for i, w in enumerate(updated_parameters)})
        out_model.seek(0)

        # Metadata needed for aggregation server side
        training_metadata = {
            "num_examples": len(X_train),
            "training_metadata": {
                "epochs": epochs,
                "batch_size": len(X_train),
                "learning_rate": model.learning_rate_init,
            },
        }

        metadata = {"training_metadata": training_metadata}
        
        train_time = time.perf_counter() - start_time

        log_entry = {"client_id": client_id,
                     "client_name": client_name,
                     "train_time": train_time,
                     "time stamp": time.time()
                     }
        

        with log_lock:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        
        return out_model, metadata
    return on_train

def on_validate(in_model):
    model = load_model(in_model)

    X_train, y_train, X_test, y_test = make_data()

    # JSON schema
    metrics = {"validation_accuracy": accuracy_score(y_test, model.predict(X_test)), "training_accuracy": accuracy_score(y_train, model.predict(X_train))}

    return metrics


def run_client(online_for=120, name="client", client_id=None):
    """Simulates a client that starts and stops
    at random intervals.

    The client will start after a random time 'mean_delay',
    stay online for 'online_for' seconds (deterministic),
    then disconnect.

    This is repeated for N_CYCLES.

    """
    if client_id is None:
        client_id = str(uuid.uuid4())

    for i in range(settings["N_CYCLES"]):
        # Sample a delay until the client starts
        t_start = np.random.randint(1, settings["CLIENTS_MAX_DELAY"])
        time.sleep(t_start)

        fl_client = FednClient(train_callback=callback_train(client_id, name), validate_callback=on_validate)
        fl_client.set_name(name)
        fl_client.set_client_id(client_id)

        controller_config = {
            "name": fl_client.name,
            "client_id": fl_client.client_id,
            "package": "local",
            "preferred_combiner": "",
        }

        url = get_api_url(host=settings["DISCOVER_HOST"], port=settings["DISCOVER_PORT"], secure=settings["SECURE"])

        result, combiner_config = fl_client.connect_to_api(url, settings["CLIENT_TOKEN"], controller_config)
        #combiner_config.host = "100.84.229.36"
        combiner_config.host = "127.0.0.1"
        fl_client.init_grpchandler(config=combiner_config, client_name=fl_client.client_id, token=settings["CLIENT_TOKEN"])

        threading.Thread(target=fl_client.run, daemon=True).start()
        time.sleep(online_for)
        fl_client.grpc_handler._disconnect()


if __name__ == "__main__":
    # We start N_CLIENTS independent client threads
    threads = []
    for i in range(settings["N_CLIENTS"]):
        time.sleep(0.01)
        t = threading.Thread(
            target=run_client,
            args=(
                settings["CLIENTS_ONLINE_FOR_SECONDS"],
                "client{}".format(i + 1),
                str(uuid.uuid4())),
                daemon=False
        )
        
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
