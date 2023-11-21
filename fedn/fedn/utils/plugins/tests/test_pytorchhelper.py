import os
import unittest

import numpy as np

from fedn.utils.plugins.pytorchhelper import Helper as PyTorchHelper


class TestPyTorchHelper(unittest.TestCase):
    """Test the PyTorchHelper class."""

    def setUp(self):
        self.helper = PyTorchHelper()

    def test_increment_average(self):
        """Test the increment_average method. The weights are stored as OrderedDicts."""

        # Model as OrderedDict with keys as torch layers and values as numpy arrays
        model = {'layer1': np.array([1, 2, 3])}
        model_next = {'layer1': np.array([4, 5, 6])}
        a = 10
        W = 20

        result = self.helper.increment_average(model, model_next, a, W)

        # Check OrderedDict values match
        np.testing.assert_array_equal(result['layer1'], np.array([2.5, 3.5, 4.5]))

        # Model as OrderedDict with keys as torch layers and values as lists
        model = {'layer1': [1, 2, 3]}
        model_next = {'layer1': [4, 5, 6]}
        a = 10
        W = 20

        # Catch TypeError: unsupported operand type(s) for -: 'list' and 'list'
        with self.assertRaises(TypeError):
            result = self.helper.increment_average(model, model_next, a, W)

    # Test save and load methods
    def test_save_load(self):
        """Test the save and load methods."""

        # Create a model
        model = {'layer1': np.array([1, 2, 3])}

        # Save the model
        self.helper.save(model, 'test_model')

        # Check if the model file exists
        self.assertTrue(os.path.exists('test_model.npz'))

        # Load the model
        result = self.helper.load('test_model.npz')

        # Check OrderedDict values match
        np.testing.assert_array_equal(result['layer1'], np.array([1, 2, 3]))

        # Remove the model file
        os.remove('test_model.npz')


if __name__ == '__main__':
    unittest.main()
