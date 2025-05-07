# Unittest for Flask API endpoints
#
# Run with:
# python -m unittest fedn.network.api.tests


import unittest
from unittest.mock import patch, MagicMock

from fedn.network.controller.control import Control
from fedn.network.storage.statestore.stores.dto.attribute import AttributeDTO
from fedn.network.storage.statestore.stores.dto.metric import MetricDTO

from fedn.network.storage.statestore.stores.dto.client import ClientDTO
from fedn.network.storage.statestore.stores.dto.combiner import CombinerDTO
from fedn.network.storage.statestore.stores.dto.model import ModelDTO
from fedn.network.storage.statestore.stores.dto.package import PackageDTO
from fedn.network.storage.statestore.stores.dto.prediction import PredictionDTO
from fedn.network.storage.statestore.stores.dto.round import RoundDTO
from fedn.network.storage.statestore.stores.dto.run import RunDTO
from fedn.network.storage.statestore.stores.dto.session import SessionDTO
from fedn.network.storage.statestore.stores.dto.status import StatusDTO
from fedn.network.storage.statestore.stores.dto.telemetry import TelemetryDTO
from fedn.network.storage.statestore.stores.dto.validation import ValidationDTO
from fedn.network.storage.statestore.stores.shared import SortOrder 

entities = ['clients', 'combiners', 'models', 'packages', 'rounds', 'sessions', 'statuses', 'validations', 'metrics', 'runs', 'predictions', 'telemetry', 'attributes']
keys = ['client_id', 'combiner_id', 'model_id', 'package_id', 'round_id', 'session_id', 'status_id', 'validation_id', 'metric_id', 'run_id', 'prediction_id', 'telemetry_id', 'attribute_id']

class MockStore:
    """Mock store implementation."""

    def get(self, id: str):
        pass

    def add(self, item):
        pass

    def update(self, item):
        pass

    def list(self, limit: int = 0, skip: int = 0, sort_key: str = None, sort_order=SortOrder.DESCENDING, **kwargs):
        pass

    def count(self, **kwargs):
        pass

class MockDB:
    """Mock database implementation."""

    def __init__(self):
        self.client_store = MockStore()
        self.validation_store = MockStore()
        self.combiner_store = MockStore()
        self.status_store = MockStore()
        self.prediction_store = MockStore()
        self.round_store = MockStore()
        self.package_store = MockStore()
        self.model_store = MockStore()
        self.session_store = MockStore()
        self.metric_store = MockStore()
        self.run_store = MockStore()
        self.prediction_store = MockStore()
        self.telemetry_store = MockStore()
        self.attribute_store = MockStore()

class NetworkAPITests(unittest.TestCase):
    """ Unittests for the Network API. """
    @patch('fedn.network.controller.controlbase.ControlBase', autospec=True)
    def setUp(self, mock_control):
        # start Flask server in testing mode
        import fedn.network.api.server
        self.app = fedn.network.api.server.app.test_client()
        self.db = MockDB()

        Control.create_instance("test_network", None, self.db)


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

    def test_get_single_endpoints(self):
        """ Test get single endpoints. """
        expected_return_id = "test"
        self.db.client_store.get = MagicMock(side_effect=lambda id: ClientDTO(client_id="test") if id == "test" else None)
        self.db.combiner_store.get = MagicMock(side_effect=lambda id: CombinerDTO(combiner_id="test") if id == "test" else None)
        self.db.model_store.get = MagicMock(side_effect=lambda id: ModelDTO(model_id="test") if id == "test" else None)
        self.db.package_store.get = MagicMock(side_effect=lambda id: PackageDTO(package_id="test") if id == "test" else None)
        self.db.round_store.get = MagicMock(side_effect=lambda id: RoundDTO(round_id="test") if id == "test" else None)
        self.db.session_store.get = MagicMock(side_effect=lambda id: SessionDTO(session_id="test") if id == "test" else None)
        self.db.status_store.get = MagicMock(side_effect=lambda id: StatusDTO(status_id="test") if id == "test" else None)
        self.db.validation_store.get = MagicMock(side_effect=lambda id: ValidationDTO(validation_id="test") if id == "test" else None)
        self.db.metric_store.get = MagicMock(side_effect=lambda id: MetricDTO(metric_id="test") if id == "test" else None)
        self.db.run_store.get = MagicMock(side_effect=lambda id: RunDTO(run_id="test") if id == "test" else None)
        self.db.prediction_store.get = MagicMock(side_effect=lambda id: PredictionDTO(prediction_id="test") if id == "test" else None)
        self.db.telemetry_store.get = MagicMock(side_effect=lambda id: TelemetryDTO(telemetry_id="test") if id == "test" else None)
        self.db.attribute_store.get = MagicMock(side_effect=lambda id: AttributeDTO(attribute_id="test") if id == "test" else None)

        for key,entity in zip(keys, entities):
            response = self.app.get(f'/api/v1/{entity}/test')
            # Assert response
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json[key], expected_return_id)
            response = self.app.get(f'/api/v1/{entity}/test2') # does not exist
            # Assert response
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json['message'], "Entity with id: test2 not found")

    def test_get_endpoints(self):
        """ Test all get endpoints. """
        excepted_return_count = 1
        expected_return_id = "test"
        self.db.client_store.list = MagicMock(return_value=[ClientDTO(client_id="test")])
        self.db.client_store.count = MagicMock(return_value=1)
        self.db.combiner_store.list = MagicMock(return_value=[CombinerDTO(combiner_id="test")])
        self.db.combiner_store.count = MagicMock(return_value=1)
        self.db.model_store.list = MagicMock(return_value=[ModelDTO(model_id="test")])
        self.db.model_store.count = MagicMock(return_value=1)
        self.db.package_store.list = MagicMock(return_value=[PackageDTO(package_id="test")])
        self.db.package_store.count = MagicMock(return_value=1)
        self.db.round_store.list = MagicMock(return_value=[RoundDTO(round_id="test")])
        self.db.round_store.count = MagicMock(return_value=1)
        self.db.session_store.list = MagicMock(return_value=[SessionDTO(session_id="test")])
        self.db.session_store.count = MagicMock(return_value=1)
        self.db.status_store.list = MagicMock(return_value=[StatusDTO(status_id="test")])
        self.db.status_store.count = MagicMock(return_value=1)
        self.db.validation_store.list = MagicMock(return_value=[ValidationDTO(validation_id="test")])
        self.db.validation_store.count = MagicMock(return_value=1)
        self.db.metric_store.list = MagicMock(return_value=[MetricDTO(metric_id="test")])
        self.db.metric_store.count = MagicMock(return_value=1)
        self.db.run_store.list = MagicMock(return_value=[RunDTO(run_id="test")])
        self.db.run_store.count = MagicMock(return_value=1)
        self.db.prediction_store.list = MagicMock(return_value=[PredictionDTO(prediction_id="test")])
        self.db.prediction_store.count = MagicMock(return_value=1)
        self.db.telemetry_store.list = MagicMock(return_value=[TelemetryDTO(telemetry_id="test")])
        self.db.telemetry_store.count = MagicMock(return_value=1)
        self.db.attribute_store.list = MagicMock(return_value=[AttributeDTO(attribute_id="test")])
        self.db.attribute_store.count = MagicMock(return_value=1)


        for key,entity in zip(keys, entities):
            response = self.app.get(f'/api/v1/{entity}/')
            # Assert response
            self.assertEqual(response.status_code, 200)

            count = response.json['count']

            self.assertEqual(count, excepted_return_count)

            id = response.json['result'][0][key]

            self.assertEqual(id, expected_return_id)

        self.db.client_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.client_store.list.assert_called_once()
        self.db.combiner_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.combiner_store.list.assert_called_once()
        self.db.model_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.model_store.list.assert_called_once()
        self.db.package_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.package_store.list.assert_called_once()
        self.db.round_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.round_store.list.assert_called_once()        
        self.db.session_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.session_store.list.assert_called_once()
        self.db.status_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.status_store.list.assert_called_once()
        self.db.validation_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.validation_store.list.assert_called_once()
        self.db.metric_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.metric_store.list.assert_called_once()
        self.db.run_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.run_store.list.assert_called_once()
        self.db.prediction_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.prediction_store.list.assert_called_once()
        self.db.telemetry_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.telemetry_store.list.assert_called_once()
        self.db.attribute_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.attribute_store.list.assert_called_once()
   
        for entity in entities:
            headers = {
                "X-Limit": 10,
                "X-Skip": 10,
                "X-Sort-Key": "test",
                "X-Sort-Order": "asc"
            }
            response = self.app.get(f'/api/v1/{entity}/', headers=headers)
            # Assert response
            self.assertEqual(response.status_code, 200)

        self.db.client_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.combiner_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.model_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.package_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.round_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)        
        self.db.session_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.status_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.validation_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.metric_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.run_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.prediction_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.telemetry_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.attribute_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)

        for entity in entities:

            response = self.app.get(f'/api/v1/{entity}/?property1=value1&property2=value2')
            # Assert response
            self.assertEqual(response.status_code, 200)

        self.db.client_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.combiner_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.model_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.package_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.round_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")        
        self.db.session_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.status_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.validation_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.metric_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.run_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.prediction_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.telemetry_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.attribute_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")

    def test_list_endpoints(self):
        """ Test all list endpoints. """
        expected_return_count = 1
        expected_return_id = "test"
        self.db.client_store.list = MagicMock(return_value=[ClientDTO(client_id="test")])
        self.db.client_store.count = MagicMock(return_value=1)
        self.db.combiner_store.list = MagicMock(return_value=[CombinerDTO(combiner_id="test")])
        self.db.combiner_store.count = MagicMock(return_value=1)
        self.db.model_store.list = MagicMock(return_value=[ModelDTO(model_id="test")])
        self.db.model_store.count = MagicMock(return_value=1)
        self.db.package_store.list = MagicMock(return_value=[PackageDTO(package_id="test")])
        self.db.package_store.count = MagicMock(return_value=1)
        self.db.round_store.list = MagicMock(return_value=[RoundDTO(round_id="test")])
        self.db.round_store.count = MagicMock(return_value=1)
        self.db.session_store.list = MagicMock(return_value=[SessionDTO(session_id="test")])
        self.db.session_store.count = MagicMock(return_value=1)
        self.db.status_store.list = MagicMock(return_value=[StatusDTO(status_id="test")])
        self.db.status_store.count = MagicMock(return_value=1)
        self.db.validation_store.list = MagicMock(return_value=[ValidationDTO(validation_id="test")])
        self.db.validation_store.count = MagicMock(return_value=1)
        self.db.metric_store.list = MagicMock(return_value=[MetricDTO(metric_id="test")])
        self.db.metric_store.count = MagicMock(return_value=1)
        self.db.run_store.list = MagicMock(return_value=[RunDTO(run_id="test")])
        self.db.run_store.count = MagicMock(return_value=1)
        self.db.prediction_store.list = MagicMock(return_value=[PredictionDTO(prediction_id="test")])
        self.db.prediction_store.count = MagicMock(return_value=1)
        self.db.telemetry_store.list = MagicMock(return_value=[TelemetryDTO(telemetry_id="test")])
        self.db.telemetry_store.count = MagicMock(return_value=1)
        self.db.attribute_store.list = MagicMock(return_value=[AttributeDTO(attribute_id="test")])
        self.db.attribute_store.count = MagicMock(return_value=1)
        
        for key,entity in zip(keys, entities):
            response = self.app.post(f'/api/v1/{entity}/list')
            # Assert response
            self.assertEqual(response.status_code, 200)

            count = response.json['count']

            self.assertEqual(count, expected_return_count)

            id = response.json['result'][0][key]

            self.assertEqual(id, expected_return_id)    

        self.db.client_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.client_store.list.assert_called_once()
        self.db.combiner_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.combiner_store.list.assert_called_once()
        self.db.model_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.model_store.list.assert_called_once()
        self.db.package_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.package_store.list.assert_called_once()
        self.db.round_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.round_store.list.assert_called_once()
        self.db.session_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.session_store.list.assert_called_once()
        self.db.status_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.status_store.list.assert_called_once()
        self.db.validation_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.validation_store.list.assert_called_once()
        self.db.metric_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.metric_store.list.assert_called_once()
        self.db.run_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.run_store.list.assert_called_once()
        self.db.prediction_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.prediction_store.list.assert_called_once()
        self.db.telemetry_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.telemetry_store.list.assert_called_once()
        self.db.attribute_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING)
        self.db.attribute_store.list.assert_called_once()

        for entity in entities:
            headers = {
                "X-Limit": 10,
                "X-Skip": 10,
                "X-Sort-Key": "test",
                "X-Sort-Order": "asc"
            }
            response = self.app.post(f'/api/v1/{entity}/list', headers=headers)
            # Assert response
            self.assertEqual(response.status_code, 200)
        self.db.client_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.combiner_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.model_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.package_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.round_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.session_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.status_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.validation_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.metric_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.run_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.prediction_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.telemetry_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)
        self.db.attribute_store.list.assert_called_with(10, 10, "test", SortOrder.ASCENDING)

        for entity in entities:
            response = self.app.post(f'/api/v1/{entity}/list', json={"property1": "value1", "property2": "value2"})
            # Assert response
            self.assertEqual(response.status_code, 200)

        self.db.client_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.combiner_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.model_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.package_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.round_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.session_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.status_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.validation_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.metric_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.run_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.prediction_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.telemetry_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")
        self.db.attribute_store.list.assert_called_with(0, 0, None, SortOrder.DESCENDING, property1="value1", property2="value2")

    def test_count_endpoints(self):
        """ Test all count endpoints. """
        expected_return_count = 1
        self.db.client_store.count = MagicMock(return_value=1)
        self.db.combiner_store.count = MagicMock(return_value=1)
        self.db.model_store.count = MagicMock(return_value=1)
        self.db.package_store.count = MagicMock(return_value=1)
        self.db.round_store.count = MagicMock(return_value=1)
        self.db.session_store.count = MagicMock(return_value=1)
        self.db.status_store.count = MagicMock(return_value=1)
        self.db.validation_store.count = MagicMock(return_value=1)
        self.db.metric_store.count = MagicMock(return_value=1)
        self.db.run_store.count = MagicMock(return_value=1)
        self.db.prediction_store.count = MagicMock(return_value=1)
        self.db.telemetry_store.count = MagicMock(return_value=1)
        self.db.attribute_store.count = MagicMock(return_value=1)

        for entity in entities:
            response = self.app.get(f'/api/v1/{entity}/count')
            # Assert response
            self.assertEqual(response.status_code, 200)

            count = response.json

            self.assertEqual(count, expected_return_count)
            
        self.db.client_store.count.assert_called_with()
        self.db.combiner_store.count.assert_called_with()
        self.db.model_store.count.assert_called_with()
        self.db.package_store.count.assert_called_with()
        self.db.round_store.count.assert_called_with()
        self.db.session_store.count.assert_called_with()
        self.db.status_store.count.assert_called_with()
        self.db.validation_store.count.assert_called_with()
        self.db.metric_store.count.assert_called_with()
        self.db.run_store.count.assert_called_with()
        self.db.prediction_store.count.assert_called_with()
        self.db.telemetry_store.count.assert_called_with()
        self.db.attribute_store.count.assert_called_with()
        
        for entity in entities:
            response = self.app.get(f'/api/v1/{entity}/count?property1=value1&property2=value2')
            # Assert response
            self.assertEqual(response.status_code, 200)
        
        self.db.client_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.combiner_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.model_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.package_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.round_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.session_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.status_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.validation_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.metric_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.run_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.prediction_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.telemetry_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.attribute_store.count.assert_called_with(property1="value1", property2="value2")
        
        for entity in entities:
            response = self.app.post(f'/api/v1/{entity}/count', json={"property1": "value1", "property2": "value2"})
            # Assert response
            self.assertEqual(response.status_code, 200)

        self.db.client_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.combiner_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.model_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.package_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.round_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.session_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.status_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.validation_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.metric_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.run_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.prediction_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.telemetry_store.count.assert_called_with(property1="value1", property2="value2")
        self.db.attribute_store.count.assert_called_with(property1="value1", property2="value2")
           

if __name__ == '__main__':
    unittest.main()
