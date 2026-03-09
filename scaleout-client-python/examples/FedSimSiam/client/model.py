import collections

import torch
import torch.nn.functional as f
from torch import nn
from torchvision.models import resnet18
from scaleoututil.utils.model import ScaleoutModel

from scaleoututil.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def D(p, z, version="simplified"):  # negative cosine similarity
    if version == "original":
        z = z.detach()  # stop gradient
        p = f.normalize(p, dim=1)  # l2-normalize
        z = f.normalize(z, dim=1)  # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == "simplified":  # same thing, much faster. Scroll down, speed test in __main__
        return - f.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class ProjectionMLP(nn.Module):
    """Projection MLP f"""

    def __init__(self, in_features, h1_features, h2_features, out_features):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features, h1_features),
            nn.BatchNorm1d(h1_features),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(h1_features, out_features),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class PredictionMLP(nn.Module):
    """Prediction MLP h"""

    def __init__(self, in_features, hidden_features, out_features):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self):
        super(SimSiam, self).__init__()
        backbone = resnet18(pretrained=False)
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

        self.backbone = backbone

        self.projector = ProjectionMLP(backbone.output_dim, 2048, 2048, 2048)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.predictor = PredictionMLP(2048, 512, 2048)

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {"loss": L}


def compile_model():
    """ Compile the pytorch model.

    :return: The compiled model.
    :rtype: torch.nn.Module
    """
    model = SimSiam()

    return model


def save_parameters(model):
    """Save model parameters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    """
    # Convert PyTorch state_dict to numpy arrays
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ScaleoutModel.from_model_params(parameters_np, helper)


def load_parameters(scaleout_model: ScaleoutModel):
    """Load model parameters from file and populate model.

    :param scaleout_model: The ScaleoutModel containing the model parameters.
    :type scaleout_model: ScaleoutModel
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    weights = scaleout_model.get_model_params(helper)
    model = compile_model()
    
    # Convert numpy arrays back to PyTorch state_dict
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = collections.OrderedDict(
        {key: torch.tensor(x) for key, x in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def init_seed(out_path="seed.npz"):
    """ Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()
    parameters_np = [val.cpu().numpy()
                     for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)

def build():
    init_seed("seed.npz")


if __name__ == "__main__":
    init_seed("../seed.npz")