import collections

import torch
import torch.nn.functional as f
from torch import nn
from torchvision.models import resnet18

from fedn.utils.helpers.helpers import get_helper

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


def save_parameters(model, out_path):
    """ Save model paramters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy()
                     for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def load_parameters(model_path):
    """ Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    model = compile_model()
    parameters_np = helper.load(model_path)

    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict(
        {key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def init_seed(out_path="seed.npz"):
    """ Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()
    save_parameters(model, out_path)


if __name__ == "__main__":
    init_seed("../seed.npz")
