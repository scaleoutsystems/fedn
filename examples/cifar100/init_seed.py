import torch
import torch.nn as nn
import torchvision.models as models

from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


# Function to replace BatchNorm layers with GroupNorm
def replace_bn_with_gn(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            replace_bn_with_gn(child, num_groups)  # Apply recursively to nested modules


def compile_model():
    # Load ResNet-18 and replace BatchNorm with GroupNorm
    resnet18 = models.resnet18(weights=None)
    replace_bn_with_gn(resnet18)
    # Modify final layer for CIFAR-100 (100 classes)
    resnet18.fc = nn.Linear(512, 100)
    return resnet18


def save_parameters(model, out_path):
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def init_seed(out_path="seed.npz"):
    model = compile_model()
    save_parameters(model, out_path)


if __name__ == "__main__":
    init_seed("seed.npz")
