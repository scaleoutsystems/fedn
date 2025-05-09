
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import click

from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"


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


def init_seed(out_path="seed.npz"):
    """Initialize seed model.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()
    save_parameters(model, out_path)


if __name__ == "__main__":
    @click.command()
    @click.argument("out_path", type=str, default="seed.npz")
    def main(out_path):
        """Initialize a seed model and save it to the specified path."""
        init_seed(out_path)

    main()
