import torch
from torch import nn

import math

import collections

from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

class NormalizerNormal(nn.Module):
    def __init__(self, mean, scale):
        super(NormalizerNormal, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean,  dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale,  dtype=torch.float32)

        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    def normalize(self, x):
        return (x-self.mean)/self.scale

    def unnormalize(self, x):
        return x*self.scale + self.mean

class NormalizerBernoulli(nn.Module):
    _epsilon  = 1e-3
    def __init__(self, mean):
        super(NormalizerBernoulli, self).__init__()
        self.register_buffer("mean", torch.minimum(torch.maximum(mean, torch.tensor(self._epsilon)), torch.tensor(1-self._epsilon)))

    def normalize_samples(self, sample):
        return sample - self.mean

    def unnormalize_logits(self, logits):
        return logits + torch.log(self.mean / (1 - self.mean))


class ConvLayerSetting:
    def __init__(self, out_channels, kernel_size, max_pool_kernel_size, activation_fn):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if max_pool_kernel_size:
            if isinstance(max_pool_kernel_size, int):
                max_pool_kernel_size = (max_pool_kernel_size, max_pool_kernel_size)
        self.max_pool_kernel_size = max_pool_kernel_size
        self.activation_fn = activation_fn


class FCLayerSetting:
    def __init__(self, out_features, dropout, activation_fn):
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.dropout = dropout




class NeuralNetworkModel(nn.Module):
    def __init__(self, in_channels, out_features, in_image_size, conv_layer_settings= (), fc_layer_settings = ()):
        super(NeuralNetworkModel, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        next_layer_in_channels = self.in_channels
        c_image_size = in_image_size

        for conv_layer_setting in conv_layer_settings:
            conv = nn.Conv2d(in_channels=next_layer_in_channels, out_channels=conv_layer_setting.out_channels,
                             kernel_size=conv_layer_setting.kernel_size, padding="same")
            self.conv_layers.append(conv)
            if conv_layer_setting.max_pool_kernel_size:
                pooling = nn.MaxPool2d(conv_layer_setting.max_pool_kernel_size, stride=conv_layer_setting.max_pool_kernel_size)
                self.conv_layers.append(pooling)
                c_image_size = (int((dim_size-1)/stride +1) for dim_size, stride in zip(c_image_size, conv_layer_setting.max_pool_kernel_size))
            self.conv_layers.append(conv_layer_setting.activation_fn)
            next_layer_in_channels = conv_layer_setting.out_channels

        self.fc_in_features = next_layer_in_channels*math.prod(c_image_size)
        next_fc_in = self.fc_in_features

        for fc_layer_setting in fc_layer_settings:
            fc = nn.Linear(next_fc_in, fc_layer_setting.out_features)
            self.fc_layers.append(fc)
            self.fc_layers.append(nn.Dropout(fc_layer_setting.dropout))
            self.fc_layers.append(fc_layer_setting.activation_fn)
            next_fc_in = fc_layer_setting.out_features

        self.final_layer = nn.Linear(next_fc_in, out_features)


    def forward(self, x):
        out = x
        out = self.conv_layers(out)
        out = out.view(-1, self.fc_in_features)
        out = self.fc_layers(out)
        out = self.final_layer(out)

        return out


class SCModel(nn.Module):
    def __init__(self, nn_model: nn.Module, label_norm: NormalizerBernoulli, spectrogram_norm: NormalizerNormal):
        super(SCModel, self).__init__()
        self.nn_model = nn_model
        self.label_norm = label_norm
        self.spectrogram_norm = spectrogram_norm

    def forward(self, spectrograms):
        spectrograms_normalized = self.spectrogram_norm.normalize(spectrograms)[:, None, ...] # Bx1xWxH

        logits_normalized = self.nn_model(spectrograms_normalized) # B x out_features

        logits = self.label_norm.unnormalize_logits(logits_normalized)

        prob = torch.nn.functional.softmax(logits, -1)

        return prob, logits



def model_hyperparams(dataset):
    n_labels = dataset.n_labels
    spectrogram_size = dataset.spectrogram_size
    label_mean, spectrogram_mean, spectrogram_std = dataset.get_stats()
    return {"n_labels":n_labels, "spectrogram_size":spectrogram_size, "label_mean":label_mean,
            "spectrogram_mean":spectrogram_mean, "spectrogram_std": spectrogram_std}


def compile_model(n_labels, spectrogram_size, label_mean, spectrogram_mean, spectrogram_std):
    spectrogram_normalizer = NormalizerNormal(spectrogram_mean, spectrogram_std)
    label_normalizer = NormalizerBernoulli(label_mean)

    conv_layers = [ConvLayerSetting(4, 3, None, nn.ReLU()), ConvLayerSetting(8, 3, 2, nn.ReLU()),
                    ConvLayerSetting(16, 3, 2, nn.ReLU()), ConvLayerSetting(32, 5, (4, 5), nn.ReLU())]

    fc_layers = [FCLayerSetting(128, 0.1, nn.ReLU()), FCLayerSetting(32, 0.1, nn.ReLU())]

    nn_model = NeuralNetworkModel(1, n_labels, spectrogram_size, conv_layers, fc_layers)
    sc_model = SCModel(nn_model, label_normalizer, spectrogram_normalizer)

    return sc_model


def load_parameters(model, parameters_stream):
    parameters_stream.seek(0)
    weights = helper.load(parameters_stream)
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

def save_parameters(model, path=None):
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return helper.save(parameters_np, path)


