import unittest
from fedn.clients.reducer.restservice import ReducerRestService
from fedn.clients.reducer.state import ReducerState
from unittest.mock import MagicMock, patch


class TestInit(unittest.TestCase):

    @patch('fedn.clients.reducer.control.ReducerControl')
    def test_discover_host(self, mock_control):
        CONFIG = {
            'discover_host': 'TEST_HOST',
            'name': 'TEST_NAME',
            'discover_port': 1111,
            'token': False,
            'remote_compute_context': True
        }
        restservice = ReducerRestService(CONFIG, mock_control, None)
        self.assertEqual(restservice.name, 'TEST_HOST')
        self.assertEqual(restservice.network_id, 'TEST_NAME-network')
        
    @patch('fedn.clients.reducer.control.ReducerControl')
    def test_name(self, mock_control):
        CONFIG = {
            'discover_host': None,
            'name': 'TEST_NAME',
            'discover_port': 1111,
            'token': False,
            'remote_compute_context': True
        }
        restservice = ReducerRestService(CONFIG, mock_control, None)
        self.assertEqual(restservice.name, 'TEST_NAME')
    
    @patch('fedn.clients.reducer.control.ReducerControl')
    def test_network_id(self, mock_control):
        CONFIG = {
            'discover_host': 'TEST_HOST',
            'name': 'TEST_NAME',
            'discover_port': 1111,
            'token': False,
            'remote_compute_context': True
        }
        restservice = ReducerRestService(CONFIG, mock_control, None)
        self.assertEqual(restservice.network_id, 'TEST_NAME-network')

class TestChecks(unittest.TestCase):
    @patch('fedn.clients.reducer.control.ReducerControl')
    def setUp(self, mock_control):
        CONFIG = {
            'discover_host': 'TEST_HOST',
            'name': 'TEST_NAME',
            'discover_port': 1111,
            'token': False,
            'remote_compute_context': True
        }

        self.restservice = ReducerRestService(CONFIG, mock_control, None)

    def test_check_compute_package(self):

        self.restservice.control.get_compute_context.return_value = {'NOT': 'NONE'}
        retval = self.restservice.check_compute_context()
        self.assertTrue(retval)

        self.restservice.control.get_compute_context.return_value = None
        retval = self.restservice.check_compute_context()
        self.assertFalse(retval)

        self.restservice.control.get_compute_context.return_value = ''
        retval = self.restservice.check_compute_context()
        self.assertFalse(retval)

        self.restservice.remote_compute_context = False 
        retval = self.restservice.check_compute_context()
        self.assertTrue(retval)

    def test_check_initial_model(self):

        self.restservice.control.get_latest_model.return_value = 'model-uid'
        retval = self.restservice.check_initial_model()
        self.assertTrue(retval)

        self.restservice.control.get_latest_model.return_value = None
        retval = self.restservice.check_initial_model()
        self.assertFalse(retval)

        self.restservice.control.get_latest_model.return_value = ''
        retval = self.restservice.check_initial_model()
        self.assertFalse(retval)


class TestToken(unittest.TestCase):

    @patch('fedn.clients.reducer.control.ReducerControl')
    def setUp(self, mock_control):
        CONFIG = {
            'discover_host': 'TEST_HOST',
            'name': 'TEST_NAME',
            'discover_port': 1111,
            'token': True,
            'remote_compute_context': True
        }

        self.restservice = ReducerRestService(CONFIG, mock_control, None)
    
    def test_encode_decode_auth_token(self):
        SECRET_KEY = 'test_secret'
        token = self.restservice.encode_auth_token(SECRET_KEY)
        payload_success = self.restservice.decode_auth_token(token, SECRET_KEY)
        payload_invalid = self.restservice.decode_auth_token('wrong_token', SECRET_KEY)
        payload_error = self.restservice.decode_auth_token(token, 'wrong_key')

        self.assertEqual(payload_success, "Success")
        self.assertEqual(payload_invalid, "Invalid token.")
        self.assertEqual(payload_error, "Invalid token.")

        



if __name__ == '__main__':
    unittest.main()