import pytest
from unittest.mock import patch
import os

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.tests.stores.helpers.mongo_docker import start_mongodb_container, stop_mongodb_container
from fedn.tests.stores.helpers.postgres_docker import start_postgres_container, stop_postgres_container


@pytest.fixture(scope="package")
def mongo_connection():
    if not os.environ.get("UNITTEST_GITHUB"):
        _, port = start_mongodb_container()
    else: 
        port = 27017

    mongo_config = {
        "type": "MongoDB",
        "mongo_config": {
            "host": "localhost",
            "port": port,
            "username": os.environ.get("UNITTEST_DBUSER", "_"),
            "password": os.environ.get("UNITTEST_DBPASS", "_"),
        }
    }
    yield DatabaseConnection(mongo_config,"test_network")

    if not os.environ.get("UNITTEST_GITHUB"):
        stop_mongodb_container()
    
@pytest.fixture(scope="package")
def sql_connection():
    sql_config = {
        "type": "SQLite",
        "sqlite_config": {
            "dbname": ":memory:",
        }
    }

    return DatabaseConnection(sql_config, "test_network")
    
@pytest.fixture(scope="package")
def postgres_connection():
    if not os.environ.get("UNITTEST_GITHUB"):
        _, port = start_postgres_container()
    else:
        port = 5432


    postgres_config={
        "type": "PostgreSQL",
        "postgres_config": {
            "username": os.environ.get("UNITTEST_DBUSER", "_"),
            "password": os.environ.get("UNITTEST_DBPASS", "_"),
            "host": "localhost",
            "port": port
        }
    }

    
    yield DatabaseConnection(postgres_config, "test_network")

    if not os.environ.get("UNITTEST_GITHUB"):
        stop_postgres_container()


@pytest.fixture(params=["postgres_connection", "sql_connection", "mongo_connection"])
def db_connection(request):
    return request.getfixturevalue(request.param)