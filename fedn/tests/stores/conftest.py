
import sys
from typing import List, Tuple
import pytest

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.tests.stores.helpers.database_helper import mongo_connection, sql_connection, postgres_connection, db_connection





@pytest.fixture(scope="package")
def all_db_connections(postgres_connection, sql_connection, mongo_connection) -> List[Tuple[str, DatabaseConnection]]:
    """
    Fixture to provide a list of database connections for testing.
    """
    return [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

# These lines ensure that pytests trigger breakpoints when assertions fail during debugging
def is_debugging():
    return 'debugpy' in sys.modules
    
# enable_stop_on_exceptions if the debugger is running during a test
if is_debugging():
  @pytest.hookimpl(tryfirst=True)
  def pytest_exception_interact(call):
    raise call.excinfo.value
    
  @pytest.hookimpl(tryfirst=True)
  def pytest_internalerror(excinfo):
    raise excinfo.value