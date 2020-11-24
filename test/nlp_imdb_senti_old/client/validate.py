import sys
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras.models as krm
from sklearn import metrics
import json
from read_data import read_data
from train import selectdata


def validate(model, data):
    print("-- RUNNING VALIDATION --", flush=True)
    try:
        (x_train, x_test, y_train, y_test, tokenizer) = read_data(data)
        model_score = model.evaluate(x_test, y_test, verbose=0)
        print('Training loss:', model_score[0])
        print('Training accuracy:', model_score[1])
        y_pred = model.predict_classes(x_test)
        clf_report = metrics.classification_report(y_test, y_pred)
    except Exception as e:
        print("failed to validate the model {}".format(e), flush=True)
        raise
    report = { 
                "classification_report": clf_report,
                "loss": model_score[0],
                "accuracy": model_score[1]
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report


if __name__ == '__main__':
    model = krm.load_model(sys.argv[1])
    report = validate(model, '../data/train.csv')
    # report = validate(model, selectdata())
    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))