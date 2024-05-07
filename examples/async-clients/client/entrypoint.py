# /bin/python
import fire
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics

HELPER_MODULE = "numpyhelper"
ARRAY_SIZE = 10000


def compile_model(max_iter=1):
    clf = MLPClassifier(max_iter=max_iter)
    # This is needed to initialize some state variables needed to make predictions
    # We will overwrite weights and biases during FL training
    X_train, y_train, _, _ = make_data()
    clf.fit(X_train, y_train)
    return clf


def save_parameters(model, out_path):
    """Save model to disk.

    :param model: The model to save.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    helper = get_helper(HELPER_MODULE)
    parameters = model.coefs_ + model.intercepts_

    helper.save(parameters, out_path)


def load_parameters(model_path):
    """Load model from disk.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    helper = get_helper(HELPER_MODULE)
    parameters = helper.load(model_path)

    return parameters


def init_seed(out_path="seed.npz"):
    """Initialize seed model.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()
    save_parameters(model, out_path)


def make_data(n_min=50, n_max=100):
    """Generate / simulate a random number n data points.

    n will fall in the interval (n_min, n_max)

    """
    n_samples = 100000
    X, y = make_classification(n_samples=n_samples, n_features=4, n_informative=4, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n = np.random.randint(n_min, n_max, 1)[0]
    ind = np.random.choice(len(X_train), n)
    X_train = X_train[ind, :]
    y_train = y_train[ind]
    return X_train, y_train, X_test, y_test


def train(in_model_path, out_model_path):
    """Train model."""
    # Load model
    parameters = load_parameters(in_model_path)
    model = compile_model()
    n = len(parameters) // 2
    model.coefs_ = parameters[:n]
    model.intercepts_ = parameters[n:]

    # Train
    X_train, y_train, _, _ = make_data()
    epochs = 10
    for i in range(epochs):
        model.partial_fit(X_train, y_train)

    # Metadata needed for aggregation server side
    metadata = {
        "num_examples": len(X_train),
    }

    # Save JSON metadata file
    save_metadata(metadata, out_model_path)

    # Save model update
    save_parameters(model, out_model_path)


def validate(in_model_path, out_json_path):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    parameters = load_parameters(in_model_path)
    model = compile_model()
    n = len(parameters) // 2
    model.coefs_ = parameters[:n]
    model.intercepts_ = parameters[n:]

    X_train, y_train, X_test, y_test = make_data()

    # JSON schema
    report = {
        "accuracy_score": accuracy_score(y_test, model.predict(X_test)),
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    fire.Fire({"init_seed": init_seed, "train": train, "validate": validate})
