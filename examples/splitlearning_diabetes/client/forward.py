import os
import sys

import torch
from data import load_data
from model import compile_model, load_client_model, save_client_model

from fedn.common.log_config import logger
from fedn.utils.helpers.helpers import get_helper, save_metadata

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

HELPER_MODULE = "splitlearninghelper"
helper = get_helper(HELPER_MODULE)

seed = 42
torch.manual_seed(seed)


def forward_pass(client_id, out_embedding_path, is_validate, data_path=None):
    """Complete a forward pass on the client side (client model) based on the local client model to produce embeddings that are sent to the combiner.

    If the forward pass is used for validation, the test dataset is loaded.

    param client_id: ID of the client to forward pass.
    :type client_id: str
    param out_embedding_path: Path to the output embedding file.
    :type out_embedding_path: str
    param is_validate: Whether to perform a validation forward pass.
    :type is_validate: bool
    param data_path: Path to the data file.
    :type data_path: str
    """
    if is_validate == "True":
        logger.info(f"Client-side validation forward pass for client {client_id}")
        X = load_data(data_path, is_train=False)
    else:
        logger.info(f"Client-side training forward pass for client {client_id}")
        X = load_data(data_path, is_train=True)

    num_local_features = X.shape[1]

    if not os.path.exists(f"{abs_path}/local_models/{client_id}.pth"):
        model = compile_model(num_local_features)
        save_client_model(model, client_id)

    model = load_client_model(client_id, num_local_features)

    model.eval()
    with torch.no_grad():
        embedding = model(X)

    # Metadata needed for aggregation server side
    metadata = {
        "num_examples": len(X),  # number of examples are mandatory
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_embedding_path)

    # save embeddings
    embedding_dict = {str(client_id): embedding.numpy()}
    helper.save(embedding_dict, out_embedding_path)


if __name__ == "__main__":
    forward_pass(sys.argv[1], sys.argv[2], sys.argv[3])  # test with: python forward.py 1 . "False" data/clients/1/diabetes.pt
