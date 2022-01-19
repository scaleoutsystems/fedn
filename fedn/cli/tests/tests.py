import pytest
from uuid import UUID
from ..run_cmd import set_token

INIT_FILE_REDUCER = {
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

def test_set_token():
    TOKEN_FLAG = "some-test-token"
    TOKEN_INIT = "fedn_token"
    
    assert set_token(INIT_FILE_REDUCER, TOKEN_FLAG) == TOKEN_FLAG
    assert set_token(INIT_FILE_REDUCER, None) == TOKEN_INIT
    assert set_token(INIT_FILE_REDUCER, "") == TOKEN_INIT
    
    COPY_INIT_FILE =  INIT_FILE_REDUCER
    del COPY_INIT_FILE["token"]
    assert set_token(COPY_INIT_FILE, TOKEN_FLAG) == TOKEN_FLAG
    
    TOKEN_UUID = set_token(COPY_INIT_FILE, None)
    uuid_obj = UUID(TOKEN_UUID, version=4)
    assert str(uuid_obj) == TOKEN_UUID

