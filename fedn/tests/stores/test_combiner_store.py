import pytest


import itertools
import datetime
import uuid

from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto import CombinerDTO
from fedn.network.storage.statestore.stores.shared import SortOrder


@pytest.fixture
def test_combiners():
    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    combiner1 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner1",
                  parent="localhost", ip="123:13:12:2", fqdn="", port=8080, address="test_address")
    combiner2 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner2",
                  parent="localhost", ip="123:13:12:2", fqdn="", port=8080, address="test_address") 
    combiner3 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner3",
                    parent="localhost", ip="123:13:12:5", fqdn="", port=8080, address="test_address")
    combiner4 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner4",
                    parent="localhost", ip="123:13:12:4", fqdn="", port=8080, address="test_address")
    combiner5 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner5",
                    parent="localhost", ip="123:13:12:3", fqdn="", port=8080, address="test_address")
    combiner6 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner6",
                    parent="localhost", ip="123:13:12:3", fqdn="", port=8080, address="test_address")
    combiner7 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner8",
                    parent="localhost", ip="123:13:12:3", fqdn="", port=8080, address="test_address")
    combiner8 = CombinerDTO(combiner_id=str(uuid.uuid4()), name="test_combiner7",
                    parent="localhost", ip="123:13:12:2", fqdn="", port=8080, address="test_address1")
    return [combiner1, combiner2, combiner3, combiner4, combiner5, combiner6, combiner7, combiner8]

@pytest.fixture
def test_combiner():
    start_date = datetime.datetime(2021, 1, 4, 1, 2, 4)
    combiner = CombinerDTO(name="test_combiner",
                  parent="localhost", ip="123:13:12:2", fqdn="", port=8080,
                  updated_at=start_date - datetime.timedelta(days=52), address="test_address")
    return combiner

@pytest.fixture
def db_connections_with_data(postgres_connection:DatabaseConnection, sql_connection: DatabaseConnection, mongo_connection:DatabaseConnection, test_combiners):
    for c in test_combiners:
        mongo_connection.combiner_store.add(c)
        postgres_connection.combiner_store.add(c)
        sql_connection.combiner_store.add(c)

    yield [("postgres", postgres_connection), ("sqlite", sql_connection), ("mongo", mongo_connection)]

    for c in test_combiners:
        mongo_connection.combiner_store.delete(c.combiner_id)
        postgres_connection.combiner_store.delete(c.combiner_id)
        sql_connection.combiner_store.delete(c.combiner_id)
    

@pytest.fixture
def options():
    sorting_keys = (None, 
                    "name",
                    "committed_at",
                    "updated_at",
                    "invalid_key") 
    limits = (None, 0, 1, 2, 99)
    skips = (None, 0, 1, 2, 99)
    desc = (None, SortOrder.DESCENDING, SortOrder.ASCENDING)
    opt_kwargs = ({}, {"ip":"123:13:12:3"}, {"address":"test_address1"}, {"fqdn":"", "name":"test_combiner77"})

    return list(itertools.product(limits, skips, sorting_keys, desc, opt_kwargs))

class TestCombinerStore:

    def test_add_update_delete(self, db_connection: DatabaseConnection, test_combiner:CombinerDTO):
        test_combiner.check_validity()

        # Add a combiner and check that we get the added combiner back
        name = test_combiner.name

        read_combiner1 = db_connection.combiner_store.add(test_combiner)
        assert isinstance(read_combiner1.combiner_id, str)
        read_combiner1_dict = read_combiner1.to_dict()
        combiner_id = read_combiner1_dict["combiner_id"]
        del read_combiner1_dict["combiner_id"]
        del read_combiner1_dict["committed_at"]
        del read_combiner1_dict["updated_at"]

        test_combiner_dict = test_combiner.to_dict()
        del test_combiner_dict["combiner_id"]
        del test_combiner_dict["committed_at"]
        del test_combiner_dict["updated_at"]

        assert read_combiner1_dict == test_combiner_dict

        # Assert we get the same combiner back
        read_combiner2 = db_connection.combiner_store.get(combiner_id)
        assert read_combiner2 is not None
        assert read_combiner2.to_dict() == read_combiner1.to_dict()

        # Assert we get the same combiner back by name
        read_combiner3 = db_connection.combiner_store.get_by_name(name)
        assert read_combiner3 is not None
        assert read_combiner3.to_dict() == read_combiner1.to_dict()

        # Delete the combiner and check that it is deleted
        success = db_connection.combiner_store.delete(combiner_id)
        assert success == True

    def test_list(self, db_connections_with_data: list[tuple[str, DatabaseConnection]], options: list[tuple]):   
        for (name1, db_1), (name2, db_2) in zip(db_connections_with_data[1:], db_connections_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:
                gathered_combiners = db_1.combiner_store.list(*opt, **kwargs)
                count =  db_1.combiner_store.count(**kwargs)
                
                gathered_combiners2 = db_2.combiner_store.list(*opt, **kwargs)
                count2 = db_2.combiner_store.count(**kwargs)

                assert(count == count2)
                assert len(gathered_combiners) == len(gathered_combiners2)

                for i in range(len(gathered_combiners)):
                    assert gathered_combiners2[i].combiner_id == gathered_combiners[i].combiner_id
                        

                        
    
