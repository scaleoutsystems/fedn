import unittest

from fedn.network.clients.client import Client


class TestClient(unittest.TestCase):
    """Test the Client class."""

    def setUp(self):
        self.client = Client()

    def test_add_grpc_metadata(self):
        """Test the _add_grpc_metadata method."""

        # Test adding metadata when it doesn't exist
        self.client._add_grpc_metadata('key1', 'value1')
        self.assertEqual(self.client.metadata, (('key1', 'value1'),))

        # Test adding metadata when it already exists
        self.client._add_grpc_metadata('key1', 'value2')
        self.assertEqual(self.client.metadata, (('key1', 'value2'),))

        # Test adding multiple metadata
        self.client._add_grpc_metadata('key2', 'value3')
        self.assertEqual(self.client.metadata, (('key1', 'value2'), ('key2', 'value3')))

        # Test adding metadata with special characters
        self.client._add_grpc_metadata('key3', 'value4!@#$%^&*()')
        self.assertEqual(self.client.metadata, (('key1', 'value2'), ('key2', 'value3'), ('key3', 'value4!@#$%^&*()')))

        # Test adding metadata with empty key
        with self.assertRaises(ValueError):
            self.client._add_grpc_metadata('', 'value5')

        # Test adding metadata with empty value
        with self.assertRaises(ValueError):
            self.client._add_grpc_metadata('key4', '')

        # Test adding metadata with None value
        with self.assertRaises(ValueError):
            self.client._add_grpc_metadata('key5', None)


if __name__ == '__main__':
    unittest.main()
