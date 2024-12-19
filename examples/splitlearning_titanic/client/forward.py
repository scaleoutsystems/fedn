import os
import sys

import numpy as np
from data import load_data
from model import compile_model, load_client_model, save_client_model

from fedn.common.log_config import logger
from fedn.utils.helpers.helpers import get_helper, save_metadata

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

HELPER_MODULE = "splitlearninghelper"
helper = get_helper(HELPER_MODULE)


def forward_pass(client_id, out_embedding_path, data_path=None):
    """Complete a forward pass on the client side (client model). Save the embeddings s.t. they can be used on
    combiner level.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).
    """
    logger.info(f"Client-side forward pass for client {client_id}")
    # Load data
    x_train = load_data(data_path, is_train=True)

    num_local_features = x_train.shape[1] # num_local_features = 2 in this case

    # check if client model exists, if not create initial client model
    if not os.path.exists(f"{abs_path}/local_models/{client_id}.pth"):
        model = compile_model(num_local_features)
        save_client_model(model, client_id)

    # Load local client model
    model = load_client_model(client_id, num_local_features)

    embedding = model(x_train)

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(x_train),
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_embedding_path)

    # save embeddings
    embedding = embedding.detach().cpu().numpy()

    embedding_dict = {str(client_id): embedding}
    helper.save(embedding_dict, out_embedding_path)

    # save embeddings locally
    if not os.path.exists(f"{abs_path}/embeddings"):
        os.makedirs(f"{abs_path}/embeddings")
    np.savez(f"{abs_path}/embeddings/embeddings_{client_id}.npz", embedding)


if __name__ == "__main__":
    forward_pass(sys.argv[1], sys.argv[2])
    # test with: python forward.py 1 . data/clients/1/titanic.pt
