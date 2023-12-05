import numpy as np

from .helperbase import HelperBase


class Helper(HelperBase):
    """ FEDn helper class for keras.Sequential. """

    def __init__(self):
        """ Initialize helper. """
        self.name = "kerashelper"
        super().__init__()

    # function to calculate an incremental weighted average of the weights
    def increment_average(self, model, model_next, num_examples, total_examples):
        """ Incremental weighted average of model weights.

        :param model: Current model weights.
        :type model: list of numpy arrays.
        :param model_next: New model weights.
        :type model_next: list of numpy arrays.
        :param num_examples: Number of examples in new model.
        :type num_examples: int
        :param total_examples: Total number of examples.
        :type total_examples: int
        :return: Incremental weighted average of model weights.
        :rtype: list of numpy arrays.
        """
        # Incremental weighted average
        w = num_examples / total_examples
        weights = []
        for i in range(len(model)):
            weights.append(w * model_next[i] + (1 - w) * model[i])

        return weights

    # function to calculate an incremental weighted average of the weights using numpy.add
    def increment_average_add(self, model, model_next, num_examples, total_examples):
        """ Incremental weighted average of model weights.

        :param model: Current model weights.
        :type model: list of numpy arrays.
        :param model_next: New model weights.
        :type model_next: list of numpy arrays.
        :param num_examples: Number of examples in new model.
        :type num_examples: int
        :param total_examples: Total number of examples.
        :type total_examples: int
        :return: Incremental weighted average of model weights.
        :rtype: list of numpy arrays.
        """
        # Incremental weighted average
        w = np.add(model, num_examples*(np.array(model_next) - np.array(model)) / total_examples)
        return w

    def save(self, weights, path=None):
        """ Serialize weights to file. The serialized model must be a single binary object.

        :param weights: List of weights in numpy format.
        :param path: Path to file.
        :return: Path to file.
        """
        if not path:
            path = self.get_tmp_path()

        weights_dict = {}
        for i, w in enumerate(weights):
            weights_dict[str(i)] = w

        np.savez_compressed(path, **weights_dict)

        return path

    def load(self, fh):
        """ Load weights from file or filelike.

        :param fh: file path, filehandle, filelike.
        :return: List of weights in numpy format.
        """
        a = np.load(fh)

        weights = []
        for i in range(len(a.files)):
            weights.append(a[str(i)])
        return weights
