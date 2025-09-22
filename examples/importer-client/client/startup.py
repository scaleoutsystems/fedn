from fedn.network.clients.fedn_client import FednClient


def startup(client: FednClient):
    MyClient(client)


class MyClient:
    def __init__(self, client: FednClient):
        self.client = client
        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)
        client.set_predict_callback(self.predict)

    def train(self, model_params, settings):
        """Train the model with the given parameters and settings."""
        # Implement training logic here
        print("Training with model parameters:", model_params)
        iterations = 100
        for i in iterations:
            # Do training
            if i % 10 == 0:
                self.client.log_metric({"training_loss": 0.1, "training_accuracy": 0.9})
            # Regularly check if the task has been aborted
            self.client.check_task_abort()  # Throws an exception if the task has been aborted
        return model_params, {"training_metadata": {"num_examples": 1}}

    def validate(self, model_params):
        """Validate the model with the given parameters."""
        # Implement validation logic here
        print("Validating with model parameters:", model_params)
        return {"validation_accuracy": 0.95}

    def predict(self, model_params, data):
        """Make predictions with the model using the given parameters and data."""
        # Implement prediction logic here
        print("Predicting with model parameters:", model_params, "and data:", data)
        return {"predictions": [1, 0, 1]}  # Example predictions
