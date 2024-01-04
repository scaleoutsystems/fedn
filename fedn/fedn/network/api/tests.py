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

import io
import time
import unittest
from unittest.mock import MagicMock, patch

import fedn


class NetworkAPITests(unittest.TestCase):
    """ Unittests for the Network API. """
    @patch('fedn.network.statestore.mongostatestore.MongoStateStore', autospec=True)
    @patch('fedn.network.controller.controlbase.ControlBase', autospec=True)
    def setUp(self, mock_mongo, mock_control):
        # start Flask server in testing mode
        import fedn.network.api.server
        self.app = fedn.network.api.server.app.test_client()

    def test_get_model_trail(self):
        """ Test get_model_trail endpoint. """
        # Mock api.get_model_trail
        model_id = "test"
        time_stamp = time.time()
        return_value = {model_id: time_stamp}
        fedn.network.api.server.api.get_model_trail = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_model_trail')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_model_trail was called
        fedn.network.api.server.api.get_model_trail.assert_called_once_with()

    def test_get_latest_model(self):
        """ Test get_latest_model endpoint. """
        # Mock api.get_latest_model
        model_id = "test"
        time_stamp = time.time()
        return_value = {model_id: time_stamp}
        fedn.network.api.server.api.get_latest_model = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_latest_model')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_latest_model was called
        fedn.network.api.server.api.get_latest_model.assert_called_once_with()

    def test_get_initial_model(self):
        """ Test get_initial_model endpoint. """
        # Mock api.get_initial_model
        model_id = "test"
        time_stamp = time.time()
        return_value = {model_id: time_stamp}
        fedn.network.api.server.api.get_initial_model = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_initial_model')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_initial_model was called
        fedn.network.api.server.api.get_initial_model.assert_called_once_with()

    def test_set_initial_model(self):
        """ Test set_initial_model endpoint. """
        # Mock api.set_initial_model
        success = True
        message = "test"
        return_value = {'success': success, 'message': message}
        fedn.network.api.server.api.set_initial_model = MagicMock(return_value=return_value)
        # Create test file
        request_file = (io.BytesIO(b"abcdef"), 'test.txt')
        # Make request
        response = self.app.post('/set_initial_model', data={"file": request_file})
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.set_initial_model was called
        fedn.network.api.server.api.set_initial_model.assert_called_once()

    def test_list_clients(self):
        """ Test list_clients endpoint. """
        # Mock api.get_all_clients
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_all_clients = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/list_clients')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_all_clients was called
        fedn.network.api.server.api.get_all_clients.assert_called_once_with()

    def test_get_active_clients(self):
        """ Test get_active_clients endpoint. """
        # Mock api.get_active_clients
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_active_clients = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_active_clients?combiner=test')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_active_clients was called
        fedn.network.api.server.api.get_active_clients.assert_called_once_with("test")

    def test_add_client(self):
        """ Test add_client endpoint. """
        # Mock api.add_client
        return_value = {"test": "test"}
        fedn.network.api.server.api.add_client = MagicMock(return_value=return_value)
        # Make request
        response = self.app.post('/add_client', json={
            'preferred_combiner': 'test',
        })
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.add_client was called
        fedn.network.api.server.api.add_client.assert_called_once_with(
            preferred_combiner="test",
            remote_addr='127.0.0.1'
        )

    def test_list_combiners(self):
        """ Test list_combiners endpoint. """
        # Mock api.get_all_combiners
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_all_combiners = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/list_combiners')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_all_combiners was called
        fedn.network.api.server.api.get_all_combiners.assert_called_once_with()

    def test_list_compute_packages(self):
        """ Test list_compute_packages endpoint. """
        # Mock api.list_compute_packages
        return_value = {"test": "test"}
        fedn.network.api.server.api.list_compute_packages = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/list_combiners')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.list_compute_packages was called
        fedn.network.api.server.api.list_compute_packages.assert_called_once_with()

    def test_list_rounds(self):
        """ Test list_rounds endpoint. """
        # Mock api.get_all_rounds
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_all_rounds = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/list_rounds')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_all_rounds was called
        fedn.network.api.server.api.get_all_rounds.assert_called_once_with()

    def test_get_round(self):
        """ Test get_round endpoint. """
        # Mock api.get_round
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_round = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_round?round_id=test')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_round was called
        fedn.network.api.server.api.get_round.assert_called_once_with("test")

    def test_get_combiner(self):
        """ Test get_combiner endpoint. """
        # Mock api.get_combiner
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_combiner = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_combiner?combiner=test')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_combiner was called
        fedn.network.api.server.api.get_combiner.assert_called_once_with("test")

    def test_add_combiner(self):
        """ Test add_combiner endpoint. """
        # Mock api.add_combiner
        success = True
        message = "test"
        return_value = {'success': success, 'message': message}
        fedn.network.api.server.api.add_combiner = MagicMock(return_value=return_value)
        # Make request
        response = self.app.post('/add_combiner', json={
            'combiner_id': 'test',
            'address': '1234',
            'port': '1234',
            'secure_grpc': 'True',
            'fqdn': 'test',
        })
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.add_combiner was called
        fedn.network.api.server.api.add_combiner.assert_called_once_with(
            combiner_id='test',
            remote_addr='127.0.0.1',
            address='1234',
            port='1234',
            secure_grpc='True',
            fqdn='test',
        )

    def test_get_events(self):
        """ Test get_events endpoint. """
        # Mock api.get_events
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_events = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_events')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_events was called
        fedn.network.api.server.api.get_events.assert_called_once()

    def test_get_status(self):
        """ Test get_status endpoint. """
        # Mock api.get_status
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_controller_status = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_controller_status')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_status was called
        fedn.network.api.server.api.get_controller_status.assert_called_once()

    def test_start_session(self):
        """ Test start_session endpoint. """
        # Mock api.start_session
        success = True
        message = "test"
        return_value = {'success': success, 'message': message}
        fedn.network.api.server.api.start_session = MagicMock(return_value=return_value)
        # Make request with only session_id
        json = {'session_id': 'test',
                'round_timeout': float(60),
                'rounds': 1,
                'round_buffer_size': -1,
                }
        response = self.app.post('/start_session', json=json)
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.start_session was called
        fedn.network.api.server.api.start_session.assert_called_once_with(
            session_id='test',
            round_timeout=float(60),
            rounds=1,
            round_buffer_size=-1,
        )

    def test_list_sessions(self):
        """ Test list_sessions endpoint. """
        # Mock api.list_sessions
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_all_sessions = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/list_sessions')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.list_sessions was called
        fedn.network.api.server.api.get_all_sessions.assert_called_once()

    def test_list_models(self):
        """ Test list_models endpoint. """
        # Mock api.list_models
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_models = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/list_models')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.list_models was called
        fedn.network.api.server.api.get_models.assert_called_once()

    def test_get_package(self):
        """ Test get_package endpoint. """
        # Mock api.get_package
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_compute_package = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_package')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_package was called
        fedn.network.api.server.api.get_compute_package.assert_called_once_with()

    def test_get_controller_status(self):
        """ Test get_controller_status endpoint. """
        # Mock api.get_controller_status
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_controller_status = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_controller_status')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_controller_status was called
        fedn.network.api.server.api.get_controller_status.assert_called_once_with()

    def test_get_client_config(self):
        """ Test get_client_config endpoint. """
        # Mock api.get_client_config
        return_value = {"test": "test"}
        fedn.network.api.server.api.get_client_config = MagicMock(return_value=return_value)
        # Make request
        response = self.app.get('/get_client_config')
        # Assert response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, return_value)
        # Assert api.get_client_config was called
        fedn.network.api.server.api.get_client_config.assert_called_once_with(True)


if __name__ == '__main__':
    unittest.main()
