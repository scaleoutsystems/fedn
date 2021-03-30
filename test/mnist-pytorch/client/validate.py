import sys
from data.read_data import read_data
import json
import yaml
import torch
import os
import collections
import pickle

def np_to_weights(weights_np):
    weights = collections.OrderedDict()
    for w in weights_np:
        weights[w] = torch.tensor(weights_np[w])
    return weights

def validate(model, settings):
    print("-- RUNNING VALIDATION --", flush=True)
    # The data, split between train and test sets. We are caching the partition in 
    # the container home dir so that the same data subset is used for 
    # each iteration.

    def evaluate(model, loss, dataloader):
        model.eval()
        train_loss = 0
        train_correct = 0
        with torch.no_grad():
            for x, y in dataloader:
                output = model(x)
                train_loss += 32 * loss(output, y).item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(y.view_as(pred)).sum().item()
            train_loss /= len(dataloader.dataset)
            train_acc = train_correct / len(dataloader.dataset)
        return float(train_loss), float(train_acc)

    # Load data
    try:
        with open('/app/mnist_data/trainset.pyb', 'rb') as fh:
            trainset = pickle.loads(fh.read())
    except:
        trainset = read_data(True, nr_examples=settings['training_samples'])
        try:
            if not os.path.isdir('/app/mnist_data'):
                os.mkdir('/app/mnist_data')
            with open('/app/mnist_data/trainset.pyb', 'wb') as fh:
                fh.write(pickle.dumps(trainset))
        except:
            pass

    try:
        with open('/app/mnist_data/testset.pyb', 'rb') as fh:
            testset = pickle.loads(fh.read())
    except:
        testset = read_data(False, nr_examples=settings['training_samples'])
        try:
            if not os.path.isdir('/app/mnist_data'):
                os.mkdir('/app/mnist_data')
            with open('/app/mnist_data/trainset.pyb', 'wb') as fh:
                fh.write(pickle.dumps(testset))
        except:
            pass

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=settings['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=settings['batch_size'], shuffle=True)

    try:
        training_loss, training_acc = evaluate(model, loss, train_loader)
        test_loss, test_acc = evaluate(model, loss, test_loader)

    except Exception as e:
        print("failed to validate the model {}".format(e), flush=True)
        raise
    
    report = { 
                "classification_report": 'unevaluated',
                "training_loss": training_loss,
                "training_accuracy": training_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report

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

    model_pack = helper.load_model(sys.argv[1])
    weights = np_to_weights(model_pack['weights'])
    k = model_pack['k']
    print("global weights k: ", k)
    model.load_state_dict(weights)

    report = validate(model, settings)

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))

