from fedn.network.clients.fedn_client import FednClient

client_reference: FednClient = None


def startup(client: FednClient):
    client_reference = client
    client.set_train_callback(train)
    client.set_validate_callback(validate)
    client.set_predict_callback(predict)


def train(model_params, settings):
    """Train the model with the given parameters and settings."""
    # Implement training logic here
    print("Training with model parameters:", model_params)
    client_reference.log_metric({"training_loss": 0.1, "training_accuracy": 0.9})
    return model_params, {"training_metadata": {"num_examples": 1}}


def validate(model_params):
    """Validate the model with the given parameters."""
    # Implement validation logic here
    print("Validating with model parameters:", model_params)
    return {"validation_accuracy": 0.95}


def predict(model_params, data):
    """Make predictions with the model using the given parameters and data."""
    # Implement prediction logic here
    print("Predicting with model parameters:", model_params, "and data:", data)
    return {"predictions": [1, 0, 1]}  # Example predictions
