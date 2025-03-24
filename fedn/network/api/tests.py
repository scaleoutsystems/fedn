# Unittest for Flask API endpoints
#
# Run with:
# python -m unittest fedn.tests.network.api.tests
#
# or
#
# python3 -m unittest fedn.tests.network.api.tests
#
# or
#
# python3 -m unittest fedn.tests.network.api.tests.NetworkAPITests
#
# or
#
# python -m unittest fedn.tests.network.api.tests.NetworkAPITests
#
# or
#
# python -m unittest fedn.tests.network.api.tests.NetworkAPITests.test_get_model_trail
#
# or
#
# python3 -m unittest fedn.tests.network.api.tests.NetworkAPITests.test_get_model_trail
#

import unittest
from unittest.mock import patch, MagicMock

import pymongo

import fedn  # noqa: F401

entitites = ['clients', 'combiners', 'models', 'packages', 'rounds', 'sessions', 'statuses', 'validations']

class NetworkAPITests(unittest.TestCase):
    """ Unittests for the Network API. """
    @patch('fedn.network.controller.controlbase.ControlBase', autospec=True)
    def setUp(self, mock_control):
        # start Flask server in testing mode
        import fedn.network.api.server
        self.app = fedn.network.api.server.app.test_client()


    def test_health(self):
        """ Test get_models endpoint. """
        response = self.app.get('/health')
        # Assert response
        self.assertEqual(response.status_code, 200)

    def test_add_combiner(self):
        """ Test get_models endpoint. """
        # Mock api.get_models
        
        response = self.app.post('/add_combiner')
        
        self.assertEqual(response.status_code, 410)

    def test_get_controller_status(self):
        """ Test get_models endpoint. """
        response = self.app.get('/get_controller_status')
        # Assert response
        self.assertEqual(response.status_code, 200)

    def test_get_endpoints(self):
        """ Test allt get endpoints. """
        return_value = {"count": 1, "results": [{"id": "test"}]}
        fedn.network.api.shared.client_store.list = MagicMock(return_value=return_value)
        fedn.network.api.shared.combiner_store.list = MagicMock(return_value=return_value)
        fedn.network.api.shared.model_store.list = MagicMock(return_value=return_value)
        fedn.network.api.shared.package_store.list = MagicMock(return_value=return_value)
        fedn.network.api.shared.round_store.list = MagicMock(return_value=return_value)
        fedn.network.api.shared.session_store.list = MagicMock(return_value=return_value)
        fedn.network.api.shared.status_store.list = MagicMock(return_value=return_value)
        fedn.network.api.shared.validation_store.list = MagicMock(return_value=return_value)

        for entity in entitites:
            response = self.app.get(f'/api/v1/{entity}/')
            # Assert response
            self.assertEqual(response.status_code, 200)

            count = response.json['count']
            expected_count = return_value['count']

            self.assertEqual(count, expected_count)

            id = response.json['results'][0]['id']
            expected_id = return_value['results'][0]['id']

            self.assertEqual(id, expected_id)

        fedn.network.api.shared.client_store.list.assert_called_with(0, 0, None, pymongo.DESCENDING)
        fedn.network.api.shared.client_store.list.assert_called_once()
        fedn.network.api.shared.combiner_store.list.assert_called_with(0, 0, None, pymongo.DESCENDING)
        fedn.network.api.shared.combiner_store.list.assert_called_once()
        fedn.network.api.shared.model_store.list.assert_called_with(0, 0, None, pymongo.DESCENDING)
        fedn.network.api.shared.model_store.list.assert_called_once()
        fedn.network.api.shared.package_store.list.assert_called_with(0, 0, None, pymongo.DESCENDING)
        fedn.network.api.shared.package_store.list.assert_called_once()
        fedn.network.api.shared.round_store.list.assert_called_with(0, 0, None, pymongo.DESCENDING)
        fedn.network.api.shared.round_store.list.assert_called_once()        
        fedn.network.api.shared.session_store.list.assert_called_with(0, 0, None, pymongo.DESCENDING)
        fedn.network.api.shared.session_store.list.assert_called_once()
        fedn.network.api.shared.status_store.list.assert_called_with(0, 0, None, pymongo.DESCENDING)
        fedn.network.api.shared.status_store.list.assert_called_once()
        fedn.network.api.shared.validation_store.list.assert_called_with(0, 0, None, pymongo.DESCENDING)
        fedn.network.api.shared.validation_store.list.assert_called_once()
   
        for entity in entitites:
            headers = {
                "X-Limit": 10,
                "X-Skip": 10,
                "X-Sort-Key": "test",
                "X-Sort-Order": "asc"
            }
            response = self.app.get(f'/api/v1/{entity}/', headers=headers)
            # Assert response
            self.assertEqual(response.status_code, 200)

        fedn.network.api.shared.client_store.list.assert_called_with(10, 10, "test", pymongo.ASCENDING)
        fedn.network.api.shared.combiner_store.list.assert_called_with(10, 10, "test", pymongo.ASCENDING)
        fedn.network.api.shared.model_store.list.assert_called_with(10, 10, "test", pymongo.ASCENDING)
        fedn.network.api.shared.package_store.list.assert_called_with(10, 10, "test", pymongo.ASCENDING)
        fedn.network.api.shared.round_store.list.assert_called_with(10, 10, "test", pymongo.ASCENDING)        
        fedn.network.api.shared.session_store.list.assert_called_with(10, 10, "test", pymongo.ASCENDING)
        fedn.network.api.shared.status_store.list.assert_called_with(10, 10, "test", pymongo.ASCENDING)
        fedn.network.api.shared.validation_store.list.assert_called_with(10, 10, "test", pymongo.ASCENDING)
           

if __name__ == '__main__':
    unittest.main()
