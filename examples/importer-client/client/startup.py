from scaleout import FednClient


def startup(client: FednClient):
    MyClient(client)


class MyClient:
    def __init__(self, client: FednClient):
        self.client = client
        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)
        client.set_predict_callback(self.predict)

        client.set_custom_callback("my_command", self.my_command)

    def train(self, model_params, settings):
        """Train the model with the given parameters and settings."""
        # Implement training logic here
        print("Training with model parameters:", model_params)
        iterations = 100
        for i in range(iterations):
            if i % 10 == 0:
                # It is possible to log metrics during training
                print(f"Training iteration {i}/{iterations}")
                self.client.log_metric("train_iteration", i)
            # Regularly check if the task has been aborted
            self.client.check_task_abort()  # Throws an exception if the task has been aborted
        # Train returns updated model parameters and {"training_metadata": {num_examples: int}, ...} 
        return model_params, {"training_metadata": {"num_examples": 1}}

    def validate(self, model_params):
        """Validate the model with the given parameters."""
        # Implement validation logic here
        print("Validating with model parameters:", model_params)
        # Return validation metrics
        return {"validation_accuracy": 0.95}

    def predict(self, model_params, data):
        """Make predictions with the model using the given parameters and data."""
        # Implement prediction logic here
        print("Predicting with model parameters:", model_params, "and data:", data)
        return {"predictions": [1, 0, 1]}  # Example predictions

    def my_command(self, command_params):
        """Handle a custom command with the given parameters."""
        print("Hello from my_command with parameters: ", command_params)
        return {"status": "custom command executed"}