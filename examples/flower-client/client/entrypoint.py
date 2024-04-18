import os

import fire
from flwr_client import app

from fedn.utils.flowercompat.client_app_adapter import FlwrClientAppAdapter
from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

flwr_adapter = FlwrClientAppAdapter(app)


def _get_node_id():
    """Get client number from environment variable."""

    number = os.environ.get("CLIENT_NUMBER", "0")
    return int(number)


def save_parameters(out_path, parameters_np):
    """Save model paramters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    helper.save(parameters_np, out_path)


def init_seed(out_path="../seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # This calls get_parameters in the flower client which needs to be implemented.
    parameters_np = flwr_adapter.init_parameters(partition_id=_get_node_id(), config={})
    save_parameters(out_path, parameters_np)


def train(in_model_path, out_model_path):
    """Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update through the flower client, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    """
    parameters_np = helper.load(in_model_path)

    # Train on flower client
    params, num_examples = flwr_adapter.train(
        parameters=parameters_np, partition_id=_get_node_id(), config={}
    )

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": num_examples,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(out_model_path, params)


def validate(in_model_path, out_json_path, data_path=None):
    """Validate model on the clients test dataset.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    parameters_np = helper.load(in_model_path)

    loss, accuracy = flwr_adapter.evaluate(parameters_np, partition_id=_get_node_id(), config={})

    # JSON schema
    report = {
        "test_loss": loss,
        "test_accuracy": accuracy,
    }
    print(f"Loss: {loss}, accuracy: {accuracy}")
    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    fire.Fire(
        {
            "init_seed": init_seed,
            "train": train,
            "validate": validate,
        }
    )
