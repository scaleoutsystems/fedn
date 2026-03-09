from scaleout import EdgeClient, ScaleoutModel
from scaleoututil.helpers.helpers import get_helper


HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

def startup(client: EdgeClient):
    MyClient(client)

class MyClient:
    def __init__(self, client: EdgeClient):
        self.client = client
        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)
        client.set_predict_callback(self.predict)

        client.set_custom_callback("my_command", self.my_command)

    def train(self, model: ScaleoutModel, settings):
        """Train the model with the given parameters and settings."""
        # Implement training logic here
        print("Training with model parameters:", model)
        model_params = model.get_model_params(helper)
        iterations = 100
        for i in range(iterations):
            if i % 10 == 0:
                # It is possible to log metrics during training
                print(f"Training iteration {i}/{iterations}")
                self.client.log_metric({"train_iteration": i})
            # Regularly check if the task has been aborted
            self.client.check_task_abort()  # Throws an exception if the task has been aborted
        # After training, return the updated model parameters and metadata
        new_model = ScaleoutModel.from_model_params(model_params, helper=helper)
        # Train returns updated model parameters and {"training_metadata": {num_examples: int}, ...} 
        return new_model, {"training_metadata": {"num_examples": 1}}

    def validate(self, model: ScaleoutModel):
        """Validate the model with the given parameters."""
        # Implement validation logic here
        model_params = model.get_model_params(helper)
        print("Validating with model parameters")
        # Return validation metrics
        return {"validation_accuracy": 0.95}

    def predict(self, model: ScaleoutModel):
        """Make predictions with the model using the given parameters and data."""
        # Implement prediction logic here
        model_params = model.get_model_params(helper)
        print("Predicting with model parameters")
        return {"predictions": [1, 0, 1]}  # Example predictions

    def my_command(self, command_params):
        """Handle a custom command with the given parameters."""
        print("Hello from my_command with parameters: ", command_params)
        return {"status": "custom command executed"}