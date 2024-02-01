
import numpy as np

from fedn.utils.helpers.helperbase import HelperBase


class Helper(HelperBase):
    """ FEDn helper class for models weights/parameters that can be transformed to numpy ndarrays. """

    def __init__(self):
        """ Initialize helper. """
        super().__init__()
        self.name = "numpyhelper"

    def increment_average(self, m1, m2, n, N):
        """ Update a weighted incremental average of model weights.

        :param m1: Current parameters.
        :type model: list of numpy ndarray
        :param m2: next parameters.
        :type model_next: list of numpy ndarray
        :param n: Number of examples used for updating m2.
        :type n: int
        :param N: Total number of examples (accumulated).
        :type N: int
        :return: Updated incremental weighted average.
        :rtype: list of numpy ndarray
        """

        return [np.add(x, n*(y-x)/N) for x, y in zip(m1, m2)]

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
