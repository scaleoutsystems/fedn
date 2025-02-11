import pytest
from unittest.mock import patch
import os

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.tests.stores.helpers.mongo_docker import start_mongodb_container, stop_mongodb_container
from fedn.tests.stores.helpers.postgres_docker import start_postgres_container, stop_postgres_container



def network_id():
    return "test_network"

@pytest.fixture(scope="package")
def mongo_connection():
    already_running, _, port = start_mongodb_container()

    def mongo_config():
        return {
            "type": "MongoDB",
            "mongo_config": {
                "host": "localhost",
                "port": port,
                "username": os.environ.get("UNITTEST_DBUSER", "_"),
                "password": os.environ.get("UNITTEST_DBPASS", "_"),
            }
        }

    with patch('fedn.network.storage.dbconnection.get_statestore_config', return_value=mongo_config()), \
         patch('fedn.network.storage.dbconnection.get_network_config', return_value=network_id()):
        yield DatabaseConnection(force_create_new=True)
    if not already_running:
        stop_mongodb_container()
    
@pytest.fixture(scope="package")
def sql_connection():
    def sql_config():
        return {
            "type": "SQLite",
            "sqlite_config": {
                "dbname": ":memory:",
            }
        }

    with patch('fedn.network.storage.dbconnection.get_statestore_config', return_value=sql_config()), \
         patch('fedn.network.storage.dbconnection.get_network_config', return_value=network_id()):
        return DatabaseConnection(force_create_new=True)
    
@pytest.fixture(scope="package")
def postgres_connection():
    already_running, _, port = start_postgres_container()
    


    def postgres_config():
        return {
            "type": "PostgreSQL",
            "postgres_config": {
                "username": os.environ.get("UNITTEST_DBUSER", "_"),
                "password": os.environ.get("UNITTEST_DBPASS", "_"),
                "database": "fedn_db",
                "host": os.environ.get("UNITTEST_POSTGRESHOST", "localhost"),
                "port": port
            }
        }

    with patch('fedn.network.storage.dbconnection.get_statestore_config', return_value=postgres_config()), \
         patch('fedn.network.storage.dbconnection.get_network_config', return_value=network_id()):
        yield DatabaseConnection(force_create_new=True)
    if not already_running:
        stop_postgres_container()