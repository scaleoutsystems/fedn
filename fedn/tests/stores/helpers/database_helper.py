import pytest
from unittest.mock import patch

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.tests.stores.helpers.mongo_docker import start_mongodb_container, stop_mongodb_container
from fedn.tests.stores.helpers.postgres_docker import start_postgres_container, stop_postgres_container


def network_id():
    return "test_network"

@pytest.fixture(scope="session")
def mongo_connection():
    _, port = start_mongodb_container()

    def mongo_config():
        return {
            "type": "MongoDB",
            "mongo_config": {
                "host": "localhost",
                "port": port,
                "username": "fedn_admin",
                "password": "password"
            }
        }

    with patch('fedn.network.storage.dbconnection.get_statestore_config', return_value=mongo_config()), \
         patch('fedn.network.storage.dbconnection.get_network_config', return_value=network_id()):
        yield DatabaseConnection(force_create_new=True)

    stop_mongodb_container()
    
@pytest.fixture(scope="session")
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
    
@pytest.fixture(scope="pack")
def postgres_connection():
    print("Starting postgres container")
    _, port = start_postgres_container()
    


    def postgres_config():
        return {
            "type": "PostgreSQL",
            "postgres_config": {
                "username": "fedn_admin",
                "password": "password",
                "database": "fedn_db",
                "host": "localhost",
                "port": port
            }
        }

    with patch('fedn.network.storage.dbconnection.get_statestore_config', return_value=postgres_config()), \
         patch('fedn.network.storage.dbconnection.get_network_config', return_value=network_id()):
        yield DatabaseConnection(force_create_new=True)
    print("Stopping postgres container")
    stop_postgres_container()