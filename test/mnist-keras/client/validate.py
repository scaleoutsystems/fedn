import sys
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from data.read_data import read_data
import pickle
import json
from sklearn import metrics
import os
import yaml
import numpy as np

def validate(model,data,settings):
    print("-- RUNNING VALIDATION --", flush=True)

    # The data, split between train and test sets. We are caching the partition in
    # the container home dir so that the same data subset is used for
    # each iteration.


    # Training error (Client validates global model on same data as it trains on.)
    try:
        x_train = np.load('/tmp/local_dataset/x_train.npz')
        y_train = np.load('/tmp/local_dataset/y_train.npz')
    except:
        (x_train, y_train, classes) = read_data(data,
                                                nr_examples=settings['training_samples'],
                                                trainset=True)
        try:
            os.mkdir('/tmp/local_dataset')
            np.save('/tmp/local_dataset/x_train.npz', x_train)
            np.save('/tmp/local_dataset/y_train.npz', y_train)

        except:
            pass

    # Test error (Client has a small dataset set aside for validation)
    try:
        x_test = np.load('/tmp/local_dataset/x_test.npz')
        y_test = np.load('/tmp/local_dataset/y_test.npz')
    except:
        (x_test, y_test, classes) = read_data(data,
                                                nr_examples=settings['test_samples'],
                                                trainset=False)
        try:
            os.mkdir('/tmp/local_dataset')
            np.save('/tmp/local_dataset/x_test.npz', x_test)
            np.save('/tmp/local_dataset/y_test.npz', y_test)

        except:
            pass


    try:
        model_score = model.evaluate(x_train, y_train, verbose=0)
        print('Training loss:', model_score[0])
        print('Training accuracy:', model_score[1])

        model_score_test = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', model_score_test[0])
        print('Test accuracy:', model_score_test[1])
        y_pred = model.predict_classes(x_test)
        clf_report = metrics.classification_report(y_test.argmax(axis=-1),y_pred)

    except Exception as e:
        print("failed to validate the model {}".format(e),flush=True)
        raise

    report = {
                "classification_report": clf_report,
                "training_loss": model_score[0],
                "training_accuracy": model_score[1],
                "test_loss": model_score_test[0],
                "test_accuracy": model_score_test[1],
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.kerashelper import KerasHelper
    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    from models.mnist_model import create_seed_model
    model = create_seed_model()
    model.set_weights(weights)

    report = validate(model,'../data/mnist.npz',settings)

    with open(sys.argv[2],"w") as fh:
        fh.write(json.dumps(report))
