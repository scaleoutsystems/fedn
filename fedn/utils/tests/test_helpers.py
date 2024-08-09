import os
import unittest

from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics


class TestHelpers(unittest.TestCase):

    def test_get_helper(self):
        helper = get_helper('pytorchhelper')

        # Check that helper is not None
        self.assertTrue(helper is not None)

        # Check that helper nane is correct
        self.assertTrue(helper.name == 'pytorchhelper')

    def test_save_metadata(self):
        metadata = {'test': 'test'}
        save_metadata(metadata, 'test')

        # Check that file exists
        self.assertTrue(os.path.exists('test-metadata'))

        # Check that file is not empty
        self.assertTrue(os.path.getsize('test-metadata') > 0)

        # Check that file contains the correct data
        with open('test-metadata', 'r') as f:
            data = f.read()
            self.assertTrue(data == '{"test": "test"}')

    def test_save_metrics(self):
        metrics = {'test': 'test'}
        save_metrics(metrics, 'test_metrics.json')

        # Check that file exists
        self.assertTrue(os.path.exists('test_metrics.json'))

        # Check that file is not empty
        self.assertTrue(os.path.getsize('test_metrics.json') > 0)

        # Check that file contains the correct data
        with open('test_metrics.json', 'r') as f:
            data = f.read()
            self.assertTrue(data == '{"test": "test"}')

    # Clean up (remove files)
    def tearDown(self):
        if os.path.exists('test-metadata'):
            os.remove('test-metadata')
        if os.path.exists('test_metrics.json'):
            os.remove('test_metrics.json')


if __name__ == '__main__':
    unittest.main()
