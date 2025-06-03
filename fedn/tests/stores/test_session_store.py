import datetime
import itertools
import time
from typing import List
import uuid

import pytest

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import ModelDTO
from fedn.network.storage.statestore.stores.dto.session import (
    SessionConfigDTO, SessionDTO)
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.tests.stores.test_store import StoreTester


@pytest.fixture
def test_model_sessions():
    model = ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model", session_id=None, name="test_name1")

    session_config = {"aggregator":"test_aggregator", "round_timeout":100, "buffer_size":100, "delete_models_storage":True, 
                      "clients_required":10, "validate":True, "helper_type":"test_helper_type", "model_id":model.model_id, "rounds": 10}

    session1 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname1",  session_config=SessionConfigDTO().patch_with(session_config))
    session2 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname2",  session_config=SessionConfigDTO().patch_with(session_config))
    session3 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname3",  session_config=SessionConfigDTO().patch_with(session_config))
    session4 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname4",  session_config=SessionConfigDTO().patch_with(session_config))
    session5 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname5",  session_config=SessionConfigDTO().patch_with(session_config))
    session6 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname6",  session_config=SessionConfigDTO().patch_with(session_config))

    return model,[session1, session2, session3, session4, session5, session6]

@pytest.fixture
def test_model_session():
    model = ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model", session_id=None, name="test_name1")

    session_config = {"aggregator":"test_aggregator", "round_timeout":100, "buffer_size":100, "delete_models_storage":True, 
                      "clients_required":10, "validate":True, "helper_type":"test_helper_type", "model_id":model.model_id, "rounds": 10}

    session = SessionDTO(name="sessionname",  session_config=SessionConfigDTO().patch_with(session_config))

    return model, session

@pytest.fixture
def options():
    sorting_keys = (None, 
                    "name",
                    "committed_at",
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"status":"test_status4"}, {"status":"blah"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))


class TestSessionStore(StoreTester):
    @pytest.fixture
    def store(self, db_connection: DatabaseConnection):
        yield db_connection.session_store
    
    @pytest.fixture
    def test_dto(self, db_connection, test_model_session: SessionDTO):
        model, test_session = test_model_session
        db_connection.model_store.add(model)
        yield test_session
        db_connection.model_store.delete(model.model_id)
        
    @pytest.fixture
    def stores_with_data(self, all_db_connections, test_model_sessions: List[SessionDTO]):
        model, test_sessions = test_model_sessions
        for connection_name, connection in all_db_connections:
            connection.model_store.add(model) 
        yield from self.helper_prepare_and_cleanup(all_db_connections, "session_store", test_sessions)
        for connection_name, connection in all_db_connections:
            connection.model_store.delete(model.model_id)

    def update_function(self, dto):
        dto.status = "new_status"
        return dto


