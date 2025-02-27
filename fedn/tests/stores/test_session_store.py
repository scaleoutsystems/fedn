import datetime
import itertools
import time
import uuid

import pymongo
import pytest

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import ModelDTO
from fedn.network.storage.statestore.stores.dto.session import (
    SessionConfigDTO, SessionDTO)


@pytest.fixture
def test_sessions():
    model = ModelDTO(model_id=str(uuid.uuid4()), parent_model="test_parent_model", session_id=None, name="test_name1")

    session_config = {"aggregator":"test_aggregator", "round_timeout":100, "buffer_size":100, "delete_models_storage":True, 
                      "clients_required":10, "validate":True, "helper_type":"test_helper_type", "model_id":model.model_id}

    session1 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname1",  session_config=SessionConfigDTO().patch(session_config))
    session2 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname2",  session_config=SessionConfigDTO().patch(session_config))
    session3 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname3",  session_config=SessionConfigDTO().patch(session_config))
    session4 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname4",  session_config=SessionConfigDTO().patch(session_config))
    session5 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname5",  session_config=SessionConfigDTO().patch(session_config))
    session6 = SessionDTO(session_id=str(uuid.uuid4()), name="sessionname6",  session_config=SessionConfigDTO().patch(session_config))

    return model,[session1, session2, session3, session4, session5, session6]

@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_sessions):
    model, test_sessions = test_sessions
    res, _ = mongo_connection.model_store.add(model)
    assert res == True
    res, _ = postgres_connection.model_store.add(model)
    assert res == True
    res, _ = sql_connection.model_store.add(model)
    assert res == True

    for c in test_sessions:
        res, _ = mongo_connection.session_store.add(c)
        assert res == True

    for c in test_sessions:
        res, _ = postgres_connection.session_store.add(c)
        assert res == True

    for c in test_sessions:
        res, _ = sql_connection.session_store.add(c)
        assert res == True

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    # TODO:Clean up



@pytest.fixture
def options():
    sorting_keys = (#None, 
                    "name",
                    #"committed_at",
                    #"invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, pymongo.DESCENDING, pymongo.ASCENDING)
    opt_kwargs = ({}, {"status":"test_status4"}, {"status":"blah"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))


class TestSessionStore:

    def test_add_update_delete(self, postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection):
        pass


    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                gathered_sessions = db_1.session_store.select(*opt, **kwargs)
                count = db_1.session_store.count(**kwargs)

                gathered_sessions2 = db_2.session_store.select(*opt, **kwargs)
                count2 = db_2.session_store.count(**kwargs)
                
                assert(len(gathered_sessions) == len(gathered_sessions2))
                assert count == count2

                for i in range(len(gathered_sessions)):
                    assert gathered_sessions2[i].name == gathered_sessions[i].name


