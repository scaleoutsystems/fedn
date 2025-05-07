from typing import List
import pytest

import datetime
import uuid
import itertools

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.attribute import AttributeDTO
from fedn.network.storage.statestore.stores.shared import SortOrder



@pytest.fixture
def test_attributes():    
    timestamp = datetime.datetime.now().replace(microsecond=0)
    

    attribute1 = AttributeDTO(attribute_id = str(uuid.uuid4()), key="data", value = "3", sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    attribute2 = AttributeDTO(attribute_id = str(uuid.uuid4()), key="data2", value="4", sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    attribute3 = AttributeDTO(attribute_id = str(uuid.uuid4()), key="data2", value="4", sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    attribute4 = AttributeDTO(attribute_id = str(uuid.uuid4()), key="data2", value="4", sender={"name":"test_sender_name1", "role":"test_sender_role1", "client_id": "test_sender_id"}, timestamp=timestamp)
    
    return [attribute1, attribute2, attribute3, attribute4]



@pytest.fixture
def test_attribute():
    timestamp = datetime.datetime.now().replace(microsecond=0)
    return AttributeDTO(key="data", value="4", sender={"name":"test_sender_name1", "role":"test_sender_role1"}, timestamp=timestamp)


@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_attributes: List[AttributeDTO]):

    for c in test_attributes:
        mongo_connection.attribute_store.add(c)
        postgres_connection.attribute_store.add(c)
        sql_connection.attribute_store.add(c)

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    for c in test_attributes:
        mongo_connection.attribute_store.delete(c.attribute_id)
        postgres_connection.attribute_store.delete(c.attribute_id)
        sql_connection.attribute_store.delete(c.attribute_id)




@pytest.fixture
def options():
    sorting_keys = (None, 
                    "sender.name", 
                    "timestamp", 
                    "invalid_key"
                    ) 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"sender.name":"test_sender_name1"}, {"key":"data"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestAttributeStore:

    def test_get_attributes_for_client(self, db_connections_with_data: List[tuple[str, DatabaseConnection]]):
        for (name1, db_1) in db_connections_with_data:
            client_id = "test_sender_id"
            attributes_distinct = db_1.attribute_store.get_current_attributes_for_client(client_id)
            attributes_all = db_1.attribute_store.list(limit=0, skip=0, sort_key="committed_at", sort_order=SortOrder.ASCENDING, **{"sender.client_id": client_id})
            attributes_distinct_2 = [attributes_all[0], attributes_all[-1]]
            assert len(attributes_distinct) == len(attributes_distinct_2)
            # assert that the attributes are equal as sets 
            assert {a.attribute_id for a in attributes_distinct} == {a.attribute_id for a in attributes_distinct_2}

            

    def test_add_update_delete(self, db_connection: DatabaseConnection, test_attribute: AttributeDTO):
        read_attribute1 = db_connection.attribute_store.add(test_attribute)
        assert isinstance(read_attribute1.attribute_id, str)
        assert isinstance(read_attribute1.committed_at, datetime.datetime)

        read_attribute1_dict = read_attribute1.to_dict()
        attribute_id = read_attribute1_dict["attribute_id"]
        del read_attribute1_dict["attribute_id"]
        del read_attribute1_dict["committed_at"]
        del read_attribute1_dict["updated_at"]

        test_attribute_dict = test_attribute.to_dict()
        del test_attribute_dict["attribute_id"]
        del test_attribute_dict["committed_at"]
        del test_attribute_dict["updated_at"]

        assert read_attribute1_dict == test_attribute_dict

        # Assert we get the same attribute back
        read_attribute2 = db_connection.attribute_store.get(attribute_id)
        assert read_attribute2 is not None
        assert read_attribute2.to_dict() == read_attribute1.to_dict()    

        # Delete the attribute and check that it is deleted
        success = db_connection.attribute_store.delete(attribute_id)
        assert success == True
    

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                gathered_models = db_1.attribute_store.list(*opt, **kwargs)
                count = db_1.attribute_store.count(**kwargs)

                gathered_models2 = db_2.attribute_store.list(*opt, **kwargs)
                count2 = db_2.attribute_store.count(**kwargs)

                assert count == count2
                assert len(gathered_models) == len(gathered_models2)

                for i in range(len(gathered_models)):
                    assert gathered_models[i].attribute_id == gathered_models2[i].attribute_id
                    
