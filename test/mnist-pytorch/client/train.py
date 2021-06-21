from __future__ import print_function
import sys
import yaml
import torch
import os
import collections
import pickle

from data.read_data import read_data

def weights_to_np(weights):

    weights_np = collections.OrderedDict()
    for w in weights:
        weights_np[w] = weights[w].cpu().detach().numpy()
    return weights_np


def np_to_weights(weights_np):
    weights = collections.OrderedDict()
    for w in weights_np:
        weights[w] = torch.tensor(weights_np[w])
    return weights


def train(model, loss, optimizer, settings):
    print("-- RUNNING TRAINING --", flush=True)

    # We are caching the partition in the container home dir so that
    # the same training subset is used for each iteration for a client. 
    try:
        with open('/tmp/local_dataset/trainset.p','rb') as fh:
            trainset = pickle.loads(fh.read())
        print("load local trainset")

    except:
        trainset = read_data(trainset=True, nr_examples=settings['training_samples'],  data_path='../data/mnist.npz')
        print("sample new local trainset")

        try:
            if not os.path.isdir('/tmp/local_dataset'):
                os.mkdir('/tmp/local_dataset')

            with open('/tmp/local_dataset/trainset.p','wb') as fh:
                fh.write(pickle.dumps(trainset))

        except:
            pass

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=settings['batch_size'], shuffle=True)

    model.train()

    for i in range(settings['epochs']):
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            error = loss(output, y)
            error.backward()
            optimizer.step()

    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.pytorchhelper import PytorchHelper
    from models.mnist_pytorch_model import create_seed_model
    helper = PytorchHelper()
    model, loss, optimizer = create_seed_model()
    model.load_state_dict(np_to_weights(helper.load_model(sys.argv[1])))
    model = train(model, loss, optimizer, settings)
    helper.save_model(weights_to_np(model.state_dict()), sys.argv[2])
