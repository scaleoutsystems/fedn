import unittest
import sys
import os
import fedn 
from unittest.mock import patch
from click.testing import CliRunner
from run_cmd import check_helper_config_file
from run_cmd import run_cmd,check_yaml_exists,logger
import click
from main import main
from fedn.network.api.server import start_server_api
from controller_cmd import main, controller_cmd

MOCK_VERSION = "0.11.1"
class TestReducerCLI(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()
        self.INIT_FILE_REDUCER = {
            "network_id": "fedn-test-network",
            "token": "fedn_token",
            "control": {
                "state": "idle",
                "helper": "keras",
            },
            "statestore": {
                "type": "MongoDB",
                "mongo_config": {
                    "username": "fedn_admin",
                    "password": "password",
                    "host": "mongo",
                    "port": "6534"
                }
            },
            "storage": {
                "storage_type": "S3",
                "storage_config": {
                    "storage_hostname": "minio",
                    "storage_port": "9000",
                    "storage_access_key": "fedn_admin",
                    "storage_secret_key": "password",
                    "storage_bucket": "fedn-models",
                    "context_bucket": "fedn-context",
                    "storage_secure_mode": "False"
                }
            }
        }

    @unittest.skip
    def test_get_statestore_config_from_file(self):
        pass

    # def test_reducer_cmd_remote(self):

    #     with self.runner.isolated_filesystem():

    #         COPY_INIT_FILE = self.INIT_FILE_REDUCER
    #         del COPY_INIT_FILE["control"]["helper"]

    #         with open('settings.yaml', 'w') as f:
    #             f.write(yaml.dump(COPY_INIT_FILE))

    #         result = self.runner.invoke(reducer_cmd, ['--remote', False, '--init',"settings.yaml"])
    #         self.assertEqual(result.output, "--remote was set to False, but no helper was found in --init settings file: settings.yaml\n")
    #         self.assertEqual(result.exit_code, -1)

    #testcase for --version in fedn 
    @patch('main.get_version')
    def test_version_output(self, mock_get_version):
        # Mock the get_version function to return a predefined version
        mock_get_version.return_value = MOCK_VERSION

        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        # Check that the command exits with a status code of 0
        self.assertEqual(result.exit_code, 0)
        
        # Check that the output contains the mocked version string
        expected_output = f"main, version {MOCK_VERSION}\n"
        self.assertEqual(result.output, expected_output)

    #train command unit test cases
    #To test check yaml function 
    @patch('run_cmd.os.path.exists')
    @patch('run_cmd.click.echo')
    def test_yaml_exists(self, mock_click_echo, mock_exists):
        path = '/fake/path'
        mock_exists.return_value = True

        # Call the function
        result = check_yaml_exists(path)
        # Assertions
        mock_exists.assert_called_once_with(os.path.join(path, 'fedn.yaml'))
        self.assertEqual(result, os.path.join(path, 'fedn.yaml'))
        mock_click_echo.assert_not_called()
    #test missing fedn yaml file
    @patch('run_cmd.os.path.exists')
    def test_missing_fedn_yaml(self, mock_exists):
        mock_exists.return_value = False
        result = self.runner.invoke(run_cmd, [
            'train',
            '--path', 'fedn/examples/mnist-pytorch/client',
            '--input', 'client.npz',
            '--output', 'client'
        ])
        self.assertEqual(result.exit_code, -1)
        self.assertIn("", result.output)

    #train cmd missing in fedn yaml file
    @patch('run_cmd._read_yaml_file')
    @patch('run_cmd.logger')
    @patch('run_cmd.exit')
    @patch('run_cmd.check_yaml_exists')
    def test_train_not_defined(self, mock_check_yaml_exists, mock_exit, mock_logger, mock_read_yaml_file):
        # Setup the mock to simulate fedn.yaml content without "train" entry point
        mock_read_yaml_file.return_value = {
            "entry_points": {
                "vaidate": "some_train_command"
            }
        }
        mock_check_yaml_exists.return_value = '/fake/path/fedn.yaml'
        result = self.runner.invoke(run_cmd, [
            'train',
            '--path', '/fake/path',
            '--input', 'input',
            '--output', 'output',
            '--remove-venv', 'True'
        ])
        mock_logger.error.assert_called_once_with("No train command defined in fedn.yaml")
        #print("hereeeeee",mock_logger.error.call_count)
        log_messages = [call[0][0] for call in mock_logger.error.call_args_list]
        #print("Captured log messages:", log_messages)
        mock_exit.assert_called_once_with(-1)

    #to test with venv flag as false
    @patch('run_cmd.os.path.exists')
    @patch('run_cmd.logger')
    @patch('run_cmd.Dispatcher')
    def test_train_cmd_with_venv_false(self, MockDispatcher,mock_exists,mock_logger):
        mock_exists.return_value = True
        mock_dispatcher = MockDispatcher.return_value
        mock_dispatcher.run_cmd.return_value = None
        result = self.runner.invoke(run_cmd, [
            'train',
            '--path', '../../.././fedn/examples/mnist-pytorch/client',
            '--input', 'client.npz',
            '--output', 'client',
            '--remove-venv', 'False'
        ])

        self.assertEqual(result.exit_code, 0)
        mock_dispatcher.run_cmd.assert_called_once_with("train client.npz client")
        #print(mock_dispatcher.run_cmd.call_count)

#Validate cmd test cases 
    @patch('run_cmd._read_yaml_file')
    @patch('run_cmd.logger')
    @patch('run_cmd.exit')
    @patch('run_cmd.check_yaml_exists')
    def test_validate_not_defined(self, mock_check_yaml_exists, mock_exit, mock_logger, mock_read_yaml_file):
        mock_read_yaml_file.return_value = {
            "entry_points": {
                "train": "some_train_command"
            }
        }
        mock_check_yaml_exists.return_value = '/fake/path/fedn.yaml'
        result = self.runner.invoke(run_cmd, [
            'validate',
            '--path', '/fake/path',
            '--input', 'input',
            '--output', 'output',
            '--remove-venv', 'True'
        ])


        # Verify that the error was logged
        mock_logger.error.assert_called_once_with("No validate command defined in fedn.yaml")
        #log_messages = [call[0][0] for call in mock_logger.error.call_args_list]
        #print("Captured log messages:", log_messages)
        mock_exit.assert_called_once_with(-1)

    #test missing fedn yaml file
    @patch('run_cmd.os.path.exists')
    def test_missing_fedn_yaml(self, mock_exists):
        mock_exists.return_value = False
        result = self.runner.invoke(run_cmd, [
            'vaidate',
            '--path', 'fedn/examples/mnist-pytorch/client',
            '--input', 'client.npz',
            '--output', 'client'
        ])
        self.assertEqual(result.exit_code, -1)
        self.assertIn("", result.output)

    #Test validate cmd with venv false
    @patch('run_cmd.os.path.exists')
    @patch('run_cmd.logger')
    @patch('run_cmd.Dispatcher')
    def test_validate_cmd_with_venv_false(self, MockDispatcher,mock_exists,mock_logger):
        mock_exists.return_value = True
        mock_dispatcher = MockDispatcher.return_value
        mock_dispatcher.run_cmd.return_value = None
        result = self.runner.invoke(run_cmd, [
            'validate',
            '--path', '../../.././fedn/examples/mnist-pytorch/client',
            '--input', 'client.npz',
            '--output', 'client',
            '--remove-venv', 'False'
        ])

        self.assertEqual(result.exit_code, 0)
        mock_dispatcher.run_cmd.assert_called_once_with("validate client.npz client")
        #print(mock_dispatcher.run_cmd.call_count)

#build cmd test cases 
    @patch('run_cmd._read_yaml_file')
    @patch('run_cmd.logger')
    @patch('run_cmd.exit')
    @patch('run_cmd.check_yaml_exists')
    def test_startup_not_defined(self, mock_check_yaml_exists, mock_exit, mock_logger, mock_read_yaml_file):
        mock_read_yaml_file.return_value = {
            "entry_points": {
                "train": "some_train_command"
            }
        }
        mock_check_yaml_exists.return_value = '/fake/path/fedn.yaml'
        runner = CliRunner()
        result = runner.invoke(run_cmd, [
            'startup',
            '--path', '/fake/path',
            '--remove-venv', 'True'
        ])


        # Verify that the error was logged
        mock_logger.error.assert_called_once_with("No startup command defined in fedn.yaml")
        log_messages = [call[0][0] for call in mock_logger.error.call_args_list]
        #print("Captured log messages:", log_messages)
        mock_exit.assert_called_once_with(-1)

    #test missing fedn yaml file
    @patch('run_cmd.os.path.exists')
    def test_missing_fedn_yaml(self, mock_exists):
        mock_exists.return_value = False
        result = self.runner.invoke(run_cmd, [
            'startup',
            '--path', 'fedn/examples/mnist-pytorch/client'
        ])
        self.assertEqual(result.exit_code, -1)
        self.assertIn("", result.output)

    @patch('run_cmd.os.path.exists')
    @patch('run_cmd.logger')
    @patch('run_cmd.Dispatcher')
    def test_startup_cmd_with_venv_false(self, MockDispatcher,mock_exists,mock_logger):
        mock_exists.return_value = True
        mock_dispatcher = MockDispatcher.return_value
        mock_dispatcher.run_cmd.return_value = None
        result = self.runner.invoke(run_cmd, [
            'startup',
            '--path', '../../.././fedn/examples/mnist-pytorch/client',
            '--remove-venv', 'False'
        ])

        self.assertEqual(result.exit_code, 0)
        mock_dispatcher.run_cmd.assert_called_once_with("startup")
        #print(mock_dispatcher.run_cmd.call_count)

    #to test controller start
    @patch('fedn.network.api.server.start_server_api')
    def test_controller_start(self, mock_start_server_api):
        runner = CliRunner()
        result = runner.invoke(main, ['controller', 'start'])
        
        # Check that the command exits with a status code of 0
        self.assertEqual(result.exit_code, 0)
        
        # Check that the start_server_api function was called
        mock_start_server_api.assert_called_once()
        #print("hereeee",mock_start_server_api.call_count)
    def test_check_helper_config_file(self):

        self.assertEqual(check_helper_config_file(
            self.INIT_FILE_REDUCER), "keras")

        COPY_INIT_FILE = self.INIT_FILE_REDUCER
        del COPY_INIT_FILE["control"]["helper"]

        with self.assertRaises(SystemExit):
            check_helper_config_file(COPY_INIT_FILE)


if __name__ == '__main__':
    unittest.main()
