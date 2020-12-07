import sys
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras as keras
import tensorflow.keras.models as krm
from read_data import read_data
import pickle
import json
from sklearn import metrics
import numpy
import os
import yaml

def validate(model,data,settings):
    print("-- RUNNING VALIDATION --", flush=True)

    # The data, split between train and test sets. We are caching the partition in 
    # the container home dir so that the same data subset is used for 
    # each iteration. 


    # Training error (Client validates global model on same data as it trains on.)
    try:
        with open('/app/mnist_train/x.pyb','rb') as fh:
            x_train=pickle.loads(fh.read())
        with open('/app/mnist_train/y.pyb','rb') as fh:
            y_train=pickle.loads(fh.read())
        with open('/app/mnist_train/classes.pyb','rb') as fh:
            classes=pickle.loads(fh.read())
    except:
        (x_train, y_train, classes) = read_data(data,nr_examples=settings['training_samples'])

        try:
            os.mkdir('/app/mnist_train')
            with open('/app/mnist_train/x.pyb','wb') as fh:
                fh.write(pickle.dumps(x_train))
            with open('/app/mnist_train/y.pyb','wb') as fh:
                fh.write(pickle.dumps(y_train))
            with open('/app/mnist_train/classes.pyb','wb') as fh:
                fh.write(pickle.dumps(classes))
        except:
            pass

    # Test error (Client has a small dataset set aside for validation)
    try:
        with open('/app/mnist_test/x_test.pyb','rb') as fh:
            x_test=pickle.loads(fh.read())
        with open('/app/mnist_test/y_test.pyb','rb') as fh:
            y_test=pickle.loads(fh.read())
        with open('/app/mnist_test/classes_test.pyb','rb') as fh:
            classes_test=pickle.loads(fh.read())
    except:
        (x_test, y_test, classes_test) = read_data("../data/test.csv",nr_examples=settings['test_samples'])

        try:
            os.mkdir('/app/mnist_test')
            with open('/app/mnist_test/x_test.pyb','wb') as fh:
                fh.write(pickle.dumps(x_test))
            with open('/app/mnist_test/y_test.pyb','wb') as fh:
                fh.write(pickle.dumps(y_test))
            with open('/app/mnist_test/classes_test.pyb','wb') as fh:
                fh.write(pickle.dumps(classes_test))
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

    print("Mattias validation starts with updates")
    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.kerasweights_modeltar import KerasWeightsHelper
    helper = KerasWeightsHelper()
    # from fedn.utils.kerassequential import KerasSequentialHelper
    #
    # helper = KerasSequentialHelper()
    model = helper.load_model(sys.argv[1])
    report = validate(model['model'],'../data/train.csv',settings)

    with open(sys.argv[2],"w") as fh:
        fh.write(json.dumps(report))
    print("Mattias validation finnish")
