import os
import unittest

import numpy as np

from fedn.utils.plugins.kerashelper import Helper as KerasHelper


class TestKerasHelper(unittest.TestCase):
    """Test the KerasHelper class."""

    def setUp(self):
        self.helper = KerasHelper()

    def test_increment_average(self):
        """Test the increment_average method."""
        # Test with a list
        model = [1, 2, 3]
        model_next = [4, 5, 6]
        a = 10
        W = 20

        result = self.helper.increment_average(model, model_next, a, W)

        self.assertEqual(result, [2.5, 3.5, 4.5])

        # Test with a numpy array
        model = np.array([1, 2, 3])
        model_next = np.array([4, 5, 6])

        result = self.helper.increment_average(model, model_next, a, W)

        np.testing.assert_array_equal(result, np.array([2.5, 3.5, 4.5]))

        # test with a list of numpy arrays
        model = [np.array([1, 2, 3])]
        model_next = [np.array([4, 5, 6])]

        result = self.helper.increment_average(model, model_next, a, W)

        np.testing.assert_array_equal(result, np.array([[2.5, 3.5, 4.5]]))

    def test_increment_average_add(self):
        """Test the increment_average_add method."""
        model = [1, 2, 3]
        model_next = [4, 5, 6]
        a = 10
        W = 20

        result = self.helper.increment_average_add(model, model_next, a, W)

        np.testing.assert_array_equal(result, np.array([2.5, 3.5, 4.5]))

        # Test with a numpy array
        model = np.array([1, 2, 3])
        model_next = np.array([4, 5, 6])

        result = self.helper.increment_average_add(model, model_next, a, W)

        np.testing.assert_array_equal(result, np.array([2.5, 3.5, 4.5]))

        # test with a list of numpy arrays
        model = [np.array([1, 2, 3])]
        model_next = [np.array([4, 5, 6])]

        result = self.helper.increment_average_add(model, model_next, a, W)

        np.testing.assert_array_equal(result, np.array([[2.5, 3.5, 4.5]]))

    def test_save(self):
        """Test the save method."""
        weights = [1, 2, 3]

        result = self.helper.save(weights, 'test.npz')

        self.assertEqual(result, 'test.npz')

    def test_load(self):
        """Test the load method."""
        weights = [1, 2, 3]

        result = self.helper.save(weights, 'test.npz')
        result = self.helper.load('test.npz')

        self.assertEqual(result, [1, 2, 3])

    # Tear down method, remove test.npz
    def tearDown(self):
        if os.path.exists('test.npz'):
            os.remove('test.npz')


if __name__ == '__main__':
    unittest.main()
