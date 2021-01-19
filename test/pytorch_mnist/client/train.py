from __future__ import print_function
import sys
import yaml
import torch

from data.read_data import read_data


def train(model, loss, optimizer, data, settings):
    print("-- RUNNING TRAINING --", flush=True)

    # We are caching the partition in the container home dir so that
    # the same training subset is used for each iteration for a client. 
    try:
        dataloader = torch.load("/app/mnist_train/dataloader.pth")
    except:
        dataloader = read_data(data, nr_examples=settings['training_samples'],
                               batch_size=settings['batch_size'])
        try:
            torch.save("/app/mnist_train/dataloader.pth")
        except:
            pass

    # model.fit(x_train, y_train, batch_size=settings['batch_size'], epochs=settings['epochs'], verbose=1)
    for i in range(settings['epochs']):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            error = loss(output, y)
            error.backward()
            optimizer.step()

    def evaluate(model, test_dl):
        model.eval()
        count = 0
        for x, y in test_dl:
            output = model(x)
            predicted = torch.max(output.data, 1)[1]
            count += (predicted == y).sum()
        return count / (len(test_dl) * test_dl.batch_size)
    print("Local accuracy: {}".format(evaluate(model, dataloader)),
          flush=True)
    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.pytorchmodel import PytorchModelHelper
    from models.mnist_pytorch_model import create_seed_model
    helper = PytorchModelHelper()
    model, loss, optimizer = create_seed_model()
    model.load_state_dict(helper.load_model(sys.argv[1]))
    model = train(model, loss, optimizer, '../data/train.csv', settings)
    helper.save_model(model.state_dict(), sys.argv[2])

