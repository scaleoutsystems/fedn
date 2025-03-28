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

from build.lib.fedn.network.storage.statestore.stores.dto.client import ClientDTO
import fedn
from fedn.network.storage.statestore.stores.dto.combiner import CombinerDTO
from fedn.network.storage.statestore.stores.dto.model import ModelDTO
from fedn.network.storage.statestore.stores.dto.package import PackageDTO
from fedn.network.storage.statestore.stores.dto.round import RoundDTO
from fedn.network.storage.statestore.stores.dto.session import SessionDTO
from fedn.network.storage.statestore.stores.dto.status import StatusDTO
from fedn.network.storage.statestore.stores.dto.validation import ValidationDTO  # noqa: F401

entitites = ['clients', 'combiners', 'models', 'packages', 'rounds', 'sessions', 'statuses', 'validations']
keys = ['client_id', 'combiner_id', 'model_id', 'package_id', 'round_id', 'session_id', 'status_id', 'validation_id']

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
        excepted_return_count = 1
        expected_return_id = "test"
        fedn.network.api.shared.client_store.list = MagicMock(return_value=[ClientDTO(client_id="test")])
        fedn.network.api.shared.client_store.count = MagicMock(return_value=1)
        fedn.network.api.shared.combiner_store.list = MagicMock(return_value=[CombinerDTO(combiner_id="test")])
        fedn.network.api.shared.combiner_store.count = MagicMock(return_value=1)
        fedn.network.api.shared.model_store.list = MagicMock(return_value=[ModelDTO(model_id="test")])
        fedn.network.api.shared.model_store.count = MagicMock(return_value=1)
        fedn.network.api.shared.package_store.list = MagicMock(return_value=[PackageDTO(package_id="test")])
        fedn.network.api.shared.package_store.count = MagicMock(return_value=1)
        fedn.network.api.shared.round_store.list = MagicMock(return_value=[RoundDTO(round_id="test")])
        fedn.network.api.shared.round_store.count = MagicMock(return_value=1)
        fedn.network.api.shared.session_store.list = MagicMock(return_value=[SessionDTO(session_id="test")])
        fedn.network.api.shared.session_store.count = MagicMock(return_value=1)
        fedn.network.api.shared.status_store.list = MagicMock(return_value=[StatusDTO(status_id="test")])
        fedn.network.api.shared.status_store.count = MagicMock(return_value=1)
        fedn.network.api.shared.validation_store.list = MagicMock(return_value=[ValidationDTO(validation_id="test")])
        fedn.network.api.shared.validation_store.count = MagicMock(return_value=1)

        for key,entity in zip(keys, entitites):
            response = self.app.get(f'/api/v1/{entity}/')
            # Assert response
            self.assertEqual(response.status_code, 200)

            count = response.json['count']

            self.assertEqual(count, excepted_return_count)

            id = response.json['result'][0][key]

            self.assertEqual(id, expected_return_id)

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
