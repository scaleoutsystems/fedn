import sys
from data.read_data import read_data
import json
import yaml
import torch

def validate(model, data, settings):
    print("-- RUNNING VALIDATION --", flush=True)
    model.eval()
    # The data, split between train and test sets. We are caching the partition in 
    # the container home dir so that the same data subset is used for 
    # each iteration. 


    # Training error (Client validates global model on same data as it trains on.)
    try:
        train_loader = torch.load("/app/mnist_train/dataloader.pth")
    except:
        train_loader = read_data(data, nr_examples=settings['training_samples'],
                               batch_size=settings['batch_size'])
        try:
            torch.save("/app/mnist_train/dataloader.pth")
        except:
            pass

    # Test error (Client has a small dataset set aside for validation)
    try:
        test_loader = torch.load("/app/mnist_test/dataloader.pth")
    except:
        test_loader = read_data(data, nr_examples=settings['test_samples'],
                               batch_size=settings['batch_size'])
        try:
            torch.save("/app/mnist_test/dataloader.pth")
        except:
            pass

    try:
        count = 0
        for x, y in train_loader:
            output = model(x)
            predicted = torch.max(output.data, 1)[1]
            count += (predicted == y).sum()
        print('Training loss:', 'unevaluated')
        train_acc = count/(len(train_loader)*train_loader.batch_size)
        print('Training accuracy: {}%'.format(100*float(train_acc)), flush=True)

        count = 0
        for x, y in test_loader:
            output = model(x)
            predicted = torch.max(output.data, 1)[1]
            count += (predicted == y).sum()
        print('Test loss:', 'unevaluated')
        test_acc = count/(len(test_loader)*test_loader.batch_size)
        print('Test accuracy: {}%'.format(100*float(test_acc)), flush=True)

    except Exception as e:
        print("failed to validate the model {}".format(e), flush=True)
        raise
    
    report = { 
                "classification_report": 'unevaluated',
                "training_loss": 'unevaluated',
                "training_accuracy": float(train_acc),
                "test_loss": 'unevaluated',
                "test_accuracy": float(test_acc),
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report

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
    report = validate(model, '../data/train.csv', settings)

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))

