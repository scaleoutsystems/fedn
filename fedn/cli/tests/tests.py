from telnetlib import COM_PORT_OPTION
import unittest
from unittest.mock import MagicMock, patch
import yaml
from click.testing import CliRunner
from uuid import UUID
from ..run_cmd import set_token, reducer_cmd, check_helper_config_file

class TestReducerCLI(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()
        self.INIT_FILE_REDUCER = {
            "network_id": "fedn-test-network",
            "token": "fedn_token",
            "control":{
                "state": "idle",
                "helper": "keras",
            },
            "statestore":{
                "type": "MongoDB",
                "mongo_config": {
                    "username": "fedn_admin",
                    "password": "password",
                    "host": "mongo",
                    "port": "6534"
                }
            },
            "storage":{
                "storage_type": "S3",
                "storage_config":{
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

    def test_set_token(self):
        TOKEN_FLAG = "some-test-token"
        TOKEN_INIT = "fedn_token"
        
        assert set_token(self.INIT_FILE_REDUCER, TOKEN_FLAG) == TOKEN_FLAG
        assert set_token(self.INIT_FILE_REDUCER, None) == TOKEN_INIT
        assert set_token(self.INIT_FILE_REDUCER, "") == TOKEN_INIT
        
        COPY_INIT_FILE =  self.INIT_FILE_REDUCER
        del COPY_INIT_FILE["token"]
        assert set_token(COPY_INIT_FILE, TOKEN_FLAG) == TOKEN_FLAG
        
        TOKEN_UUID = set_token(COPY_INIT_FILE, None)
        uuid_obj = UUID(TOKEN_UUID, version=4)
        assert str(uuid_obj) == TOKEN_UUID
    
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

    def test_check_helper_config_file(self):
        
        self.assertEqual(check_helper_config_file(self.INIT_FILE_REDUCER), "keras")
        
        COPY_INIT_FILE = self.INIT_FILE_REDUCER
        del COPY_INIT_FILE["control"]["helper"]
        
        with self.assertRaises(SystemExit):
            helper = check_helper_config_file(COPY_INIT_FILE)
        

if __name__ == '__main__':
    unittest.main()