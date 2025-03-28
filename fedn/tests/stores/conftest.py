
import sys
import pytest

from fedn.tests.stores.helpers.database_helper import mongo_connection, sql_connection, postgres_connection, db_connection


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