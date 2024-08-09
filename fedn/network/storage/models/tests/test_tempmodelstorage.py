import os
import unittest
from unittest.mock import MagicMock, patch

from fedn.common.storage.models.tempmodelstorage import TempModelStorage


class TestTempModelStorage(unittest.TestCase):

    def setUp(self):
        # Setup mock for os.environ.get for FEDN_MODEL_DIR
        self.patcher = patch('os.environ.get')
        self.mock_get = self.patcher.start()
        # Return value of mock should same folder as this file
        self.mock_get.return_value = os.path.dirname(os.path.realpath(__file__))

        # Setup storage
        self.storage = TempModelStorage()

        # add mock data to storage dicts
        self.storage.models = {"model_id1": "model1", "model_id2": "model2"}
        self.storage.models_metadata = {"model_id1": "model1", "model_id2": "model2"}

        # Create mock file as BytesIO object
        self.mock_file = MagicMock()
        self.mock_file.read.return_value = "model1"
        self.mock_file.seek.return_value = 0
        self.mock_file.write.return_value = None

    # Test that the storage is initialized with the correct default directory and data structures
    def test_init(self):
        self.assertEqual(self.storage.default_dir, os.path.dirname(os.path.realpath(__file__)))
        self.assertEqual(self.storage.models, {"model_id1": "model1", "model_id2": "model2"})
        self.assertEqual(self.storage.models_metadata, {"model_id1": "model1", "model_id2": "model2"})

    # Test that the storage can get a model

    def test_get(self):
        """ Test that the storage can get a model """

        # Test that it returns None if model_id does not exist
        self.assertEqual(self.storage.get("model_id3"), None)

        # TODO: Patch fedn.ModelStatus.OK and open to return True and mock_file respectively

    def test_get_metadata(self):
        """ Test that the storage can get a model metadata """

        # Test that it returns KeyError if model_id does not exist
        with self.assertRaises(KeyError):
            self.storage.get_meta("model_id3")

        # Test that it returns the correct metadata if model_id exists
        self.assertEqual(self.storage.get_meta("model_id1"), "model1")

    def test_set_meta(self):
        """ Test that the storage can set a model metadata """

        # Test that it returns the correct metadata if model_id exists
        self.storage.set_meta("model_id1", "model3")
        self.assertEqual(self.storage.get_meta("model_id1"), "model3")

    def test_delete(self):
        """ Test that the storage can delete a model """

        # Test that it returns False if model_id does not exist
        self.assertEqual(self.storage.delete("model_id3"), False)

        # Patch os.remove to return True
        with patch('os.remove', return_value=True) as mock_remove:

            # Test that it returns True if model_id exists
            self.assertEqual(self.storage.delete("model_id1"), True)

            # Test that os.remove is called with the correct path
            mock_remove.assert_called_with(os.path.join(self.storage.default_dir, "model_id1"))

            # Test that the model is removed from the storage
            self.assertEqual(self.storage.models, {"model_id2": "model2"})

            # Test that the model metadata is removed from the storage
            self.assertEqual(self.storage.models_metadata, {"model_id2": "model2"})

    def test_delete_all(self):
        """ Test that the storage can delete all models """

        # Patch os.remove to return True
        with patch('os.remove', return_value=True) as mock_remove:

            # Test that it returns True if model_id exists
            self.assertEqual(self.storage.delete_all(), True)

            # Test that os.remove is called with the correct path
            mock_remove.assert_called_with(os.path.join(self.storage.default_dir, "model_id2"))

            # Test that the model is removed from the storage
            self.assertEqual(self.storage.models, {})

            # Test that the model metadata is removed from the storage
            self.assertEqual(self.storage.models_metadata, {})


if __name__ == '__main__':
    unittest.main()
