import torch
import torch.nn.functional as f
from torch_geometric.nn import GATConv
import collections

from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

def compile_model():
    """Compile the graph neural network model.

    :return: The compiled model.
    :rtype: torch.nn.Module
    """

    class GAT(torch.nn.Module):
        def __init__(self, in_channels=10, out_channels=2, num_heads=4, dropout=0.2):
            super(GAT, self).__init__()
            self.dropout = dropout
            self.conv1 = GATConv(in_channels, out_channels, heads=num_heads, concat=False, dropout=dropout)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            # Apply dropout
            x = f.dropout(x, p=self.dropout, training=self.training)

            # Apply the GAT convolution layer
            x = self.conv1(x, edge_index)
            x = f.relu(x)
            return x

    return GAT()


def save_parameters(model, out_path):
    """Save model paramters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def load_parameters(model_path):
    """Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    model = compile_model()
    parameters_np = helper.load(model_path)

    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def init_seed(out_path="seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()
    print(model)
    save_parameters(model, out_path)


if __name__ == "__main__":
    init_seed("../seed.npz")
