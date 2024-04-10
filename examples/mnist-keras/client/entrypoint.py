import json
import os

import fire
import numpy as np
import tensorflow as tf

from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics

HELPER_MODULE = 'numpyhelper'
helper = get_helper(HELPER_MODULE)

NUM_CLASSES = 10

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def _get_data_path():
    data_path = os.environ.get('FEDN_DATA_PATH', abs_path + '/data/clients/1/mnist.npz')

    return data_path


def compile_model(img_rows=28, img_cols=28):
    """ Compile the TF model.

    param: img_rows: The number of rows in the image
    type: img_rows: int
    param: img_cols: The number of rows in the image
    type: img_cols: int
    return: The compiled model
    type: keras.model.Sequential
    """
    # Set input shape
    input_shape = (img_rows, img_cols, 1)

    # Define model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


def load_data(data_path, is_train=True):
    # Load data
    if data_path is None:
        data = np.load(_get_data_path())
    else:
        data = np.load(data_path)

    if is_train:
        X = data['x_train']
        y = data['y_train']
    else:
        X = data['x_test']
        y = data['y_test']

    # Normalize
    X = X.astype('float32')
    X = np.expand_dims(X, -1)
    X = X / 255
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)

    return X, y


def init_seed(out_path='../seed.npz'):
    """ Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    weights = compile_model().get_weights()
    helper.save(weights, out_path)


def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1):
    """ Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    """
    # Load data
    x_train, y_train = load_data(data_path)

    # Load model
    model = compile_model()
    weights = helper.load(in_model_path)
    model.set_weights(weights)

    # Train
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        'num_examples': len(x_train),
        'batch_size': batch_size,
        'epochs': epochs,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    weights = model.get_weights()
    helper.save(weights, out_model_path)


def validate(in_model_path, out_json_path, data_path=None):
    """ Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """

    # Load data
    x_train, y_train = load_data(data_path)
    x_test, y_test = load_data(data_path, is_train=False)

    # Load model
    model = compile_model()
    helper = get_helper(HELPER_MODULE)
    weights = helper.load(in_model_path)
    model.set_weights(weights)

    # Evaluate
    model_score = model.evaluate(x_train, y_train)
    model_score_test = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    # JSON schema
    report = {
        "training_loss": model_score[0],
        "training_accuracy": model_score[1],
        "test_loss": model_score_test[0],
        "test_accuracy": model_score_test[1],
    }

    # Save JSON
    save_metrics(report, out_json_path)


def predict(in_model_path, out_json_path, data_path=None):
    # Using test data for inference but another dataset could be loaded
    x_test, _ = load_data(data_path, is_train=False)

    # Load model
    model = compile_model()
    helper = get_helper(HELPER_MODULE)
    weights = helper.load(in_model_path)
    model.set_weights(weights)

    # Infer
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Save JSON
    with open(out_json_path, "w") as fh:
        fh.write(json.dumps({'predictions': y_pred.tolist()}))


if __name__ == '__main__':
    fire.Fire({
        'init_seed': init_seed,
        'train': train,
        'validate': validate,
        'predict': predict,
        '_get_data_path': _get_data_path,  # for testing
    })
