from model import load_parameters, save_parameters
from train import train
from validate import validate
from fedn import FednClient
from fedn.network.clients.fedn_client import ConnectToApiResult
from fedn.utils.helpers.helpers import get_helper
from io import BytesIO
import json
import os
import uuid
from config import settings

helper = get_helper("numpyhelper")

def get_api_url(api_url: str, api_port: int, secure: bool = False):
    if secure:
        url = f"https://{api_url}:{api_port}" if api_port else f"https://{api_url}"
    else:
        url = f"http://{api_url}:{api_port}" if api_port else f"http://{api_url}"
    if not url.endswith("/"):
        url += "/"
    return url

def on_train(in_model, client_settings):

    # Save model to temp file
    inpath = helper.get_tmp_path()
    with open(inpath, "wb") as fh:
        fh.write(in_model.getbuffer())

    # Train
    outpath = helper.get_tmp_path()
    train(inpath, outpath, data_path=None, batch_size=32, epochs=1, lr=0.01)

    # Load model
    model = load_parameters(outpath)

    # Serialize model
    out_model = BytesIO()
    save_parameters(model, out_model)
    out_model.seek(0)

    # Return model and metadata
    training_metadata = {
        "num_examples": 1000,
        "batch_size": 32,
        "epochs": 1,
        "lr": 0.01,
    }
    metadata = {"training_metadata": training_metadata}

    os.unlink(outpath)
    os.unlink(inpath)

    return out_model, metadata

def on_validate(in_model):
    # Save model to temp file
    inpath = helper.get_tmp_path()
    with open(inpath, "wb") as fh:
        fh.write(in_model.getbuffer())

    # Validate
    outpath = helper.get_tmp_path()
    validate(inpath, outpath, data_path=None)

    # Load metrics
    with open(outpath, "r") as fh:
        metrics = json.load(fh)

    os.unlink(outpath)
    os.unlink(inpath)

    return metrics

if __name__ == "__main__":
    client = FednClient(train_callback=on_train, validate_callback=on_validate)
    url = get_api_url(settings["DISCOVER_HOST"], settings["DISCOVER_PORT"], settings["SECURE"])
    client_id = str(uuid.uuid4())
    client.set_name(f"mnist-pytorch-client-{client_id[:4]}")
    client.set_client_id(client_id)

    controller_config = {
        "name": client.name,
        "client_id": client.client_id,
        "package": "local",
        "preferred_combiner": "",
    }

    result, combiner_config = client.connect_to_api(url=url, token=settings["CLIENT_TOKEN"], json=controller_config)

    if result != ConnectToApiResult.Assigned:
        print("Failed to connect to API, exiting.")
        exit(1)

    result = client.init_grpchandler(config=combiner_config, client_name=client.client_id, token=settings["CLIENT_TOKEN"])

    if not result:
        print("Failed to initialize gRPC handler, exiting.")
        exit(1)

    client.run()