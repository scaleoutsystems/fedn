import io
import os
import sys

import numpy as np
from data import load_data
from model import load_parameters, save_parameters

from fedn.network.clients.fedn_client import FednClient

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def startup(client: FednClient):
    MyClient(client)


class MyClient:
    def __init__(self, client: FednClient):
        self.client = client
        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)
        client.set_predict_callback(self.predict)

    def train(self, in_model_path, settings, data_path=None, batch_size=32, epochs=5):
        """Complete a model update.

        Load model paramters from in_model_path (managed by the FEDn client),
        perform a model update, and write updated paramters
        to out_model_path (picked up by the FEDn client).

        :param in_model_path: The path to the input model.
        :type in_model_path: str
        :param settings: currently unused
        :type settings: dict
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
        model = load_parameters(in_model_path)

        # Train
        for epoch in range(epochs):
            print("epoch: ", epoch, "/", epochs)
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
            self.client.log_metric({"training_loss": history.history["loss"][-1], "training_accuracy": history.history["accuracy"][-1]})

        # Metadata needed for aggregation server side
        metadata = {
            # num_examples are mandatory
            "num_examples": len(x_train),
            "batch_size": batch_size,
            "epochs": epochs,
        }
        # Regularly check if the task has been aborted
        self.client.check_task_abort()  # Throws an exception if the task has been aborted
        # Save model update (mandatory)
        upd_parameters = save_parameters(model, io.BytesIO())
        return upd_parameters, {"training_metadata": metadata}

    def validate(self, in_model_path, data_path=None):
        """Validate model.

        :param in_model_path: The path to the input model.
        :type in_model_path: str
        :param data_path: The path to the data file.
        :type data_path: str
        """
        # Load data
        x_train, y_train = load_data(data_path)
        x_test, y_test = load_data(data_path, is_train=False)

        # Load model
        model = load_parameters(in_model_path)

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

        return report

    def predict(self, in_model_path, data_path=None):
        # Using test data for prediction but another dataset could be loaded
        x_test, _ = load_data(data_path, is_train=False)

        # Load model
        model = load_parameters(in_model_path)

        # Predict
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)

        return {"predictions": y_pred}
