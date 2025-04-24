import asyncio
import random
import time
import uuid

import numpy as np
from io import BytesIO
from sklearn.metrics import accuracy_score

from init_seed import compile_model, make_data
from config import settings
from fedn import FednClient

HELPER_MODULE = "numpyhelper"

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

def on_train(in_model, client_settings):
    print("Running training callback...")
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

    return out_model, metadata

def on_validate(in_model):
    model = load_model(in_model)

    X_train, y_train, X_test, y_test = make_data()

    # JSON schema
    metrics = {"validation_accuracy": accuracy_score(y_test, model.predict(X_test)), "training_accuracy": accuracy_score(y_train, model.predict(X_train))}

    return metrics

async def async_run_fedn(fl_client):
    """Run fl_client.run() in a thread to avoid blocking event loop"""
    await asyncio.to_thread(fl_client.run)

async def simulated_client(client_index): 
    """
    Simulate one client with random offline-online intervals with one process under asyncio
    """
    client_id = str(uuid.uuid4())
    name = f"client{client_index + 1}"

    for i in range(settings["N_CYCLES"]):
        # Sample a delay until the client starts
        t_start = np.random.randint(1, settings["CLIENTS_MAX_DELAY"])
        await asyncio.sleep(t_start)

        fl_client = FednClient(train_callback=on_train, validate_callback=on_validate)
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
        combiner_config.host = "100.84.229.36"
        fl_client.init_grpchandler(config=combiner_config, 
                                   client_name=fl_client.client_id, 
                                   token=settings["CLIENT_TOKEN"])

        fedn_task = asyncio.create_task(async_run_fedn(fl_client))

        online_for = settings["CLIENTS_ONLINE_FOR_SECONDS"]
        await asyncio.sleep(online_for)


        fl_client.grpc_handler._disconnect()
        print(f"{name} Disconnected, after online {online_for}")

        await fedn_task
    
    print(f"{name} All cycles finished")

async def main():
    N = settings["N_CLIENTS"]
    tasks = [asyncio.create_task(simulated_client(i)) for i in range(N)]
    print("debug tasks: ", tasks)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())