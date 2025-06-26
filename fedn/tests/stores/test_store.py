

import datetime
import inspect
from typing import List, Tuple
import pytest
from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.statestore.stores.dto.shared import BaseDTO
from fedn.network.storage.statestore.stores.store import Store



class StoreTester:

    def update_function(self, dto: BaseDTO) -> BaseDTO:
        """Function to update the DTO. This should be overridden in the subclass."""
        raise NotImplementedError("Subclasses should implement this method or disable the test.")    

    def test_add_delete(self, store:Store, test_dto: BaseDTO):
        added_dto: BaseDTO = store.add(test_dto)
        added_primary_id = added_dto.primary_id
        assert added_dto is not test_dto, "Added DTO should be a new instance, not the original test DTO."
        assert isinstance(added_dto.primary_id, str)
        assert isinstance(added_dto.committed_at, datetime.datetime)
        assert isinstance(added_dto.updated_at, datetime.datetime)

        added_dto_dict = added_dto.to_dict()
        
        del added_dto_dict[added_dto.primary_key]
        del added_dto_dict["committed_at"]
        del added_dto_dict["updated_at"]

        test_dto_dict = test_dto.to_dict()
        del test_dto_dict[added_dto.primary_key]
        del test_dto_dict["committed_at"]
        del test_dto_dict["updated_at"]

        assert added_dto_dict == test_dto_dict 

        # Delete the attribute and check that it is deleted
        success = store.delete(added_primary_id)
        assert success == True

    def test_get(self, store: Store, test_dto: BaseDTO):
        added_dto: BaseDTO = store.add(test_dto)

        # Assert we get the same attribute back
        get_dto: BaseDTO = store.get(added_dto.primary_id)
        assert get_dto is not None
        assert get_dto.to_dict() == added_dto.to_dict()    

        store.delete(added_dto.primary_id)

    def test_update(self, store:Store, test_dto: BaseDTO):
        added_dto: BaseDTO = store.add(test_dto)
        added_primary_id = added_dto.primary_id
        
        # Update the entity and check that we get the updated entity back
        if inspect.isgeneratorfunction(self.update_function):
            for altered_dto in self.update_function(added_dto):
                updated_dto = store.update(altered_dto)
                assert updated_dto is not altered_dto, "Updated DTO should be a new instance, not the original added DTO."
                updated_dto_dict = updated_dto.to_dict()
                del updated_dto_dict["updated_at"]

                altered_dto_dict = altered_dto.to_dict()
                del altered_dto_dict["updated_at"]
                assert updated_dto_dict == altered_dto_dict

                # Assert that get returns the update entity
                get_updated_dto: BaseDTO = store.get(added_primary_id)
                assert get_updated_dto is not None
                assert get_updated_dto.to_dict() == updated_dto.to_dict()
        else:
            altered_dto = self.update_function(added_dto)
            updated_dto = store.update(altered_dto)
            assert updated_dto is not altered_dto, "Updated DTO should be a new instance, not the original added DTO."
            updated_dto_dict = updated_dto.to_dict()
            del updated_dto_dict["updated_at"]

            altered_dto_dict = altered_dto.to_dict()
            del altered_dto_dict["updated_at"]
            assert updated_dto_dict == altered_dto_dict

            # Assert that get returns the update entity
            get_updated_dto: BaseDTO = store.get(added_primary_id)
            assert get_updated_dto is not None
            assert get_updated_dto.to_dict() == updated_dto.to_dict()

        store.delete(added_primary_id)
        
    def helper_prepare_and_cleanup(self, connections: List[Tuple[str, DatabaseConnection]], store_name:str, items: List[BaseDTO]):
        for c in items:
            for _, connection in connections:
                store:Store = getattr(connection, store_name)
                store.add(c)

        yield [(connection_name, getattr(connection, store_name)) for connection_name, connection in connections]

        for c in items:
            for _, connection in connections:
                store:Store = getattr(connection, store_name)
                store.delete(c.primary_id)


    def test_list(self, stores_with_data: DatabaseConnection, options):
         for (name1, store_1), (name2, store_2) in zip(stores_with_data[1:], stores_with_data[:-1]):
            print("Running tests between databases {} and {}".format(name1, name2))
            for *opt,kwargs in options:

                fetched_records_1: List[BaseDTO] = store_1.list(*opt, **kwargs)
                count_1 = store_1.count(**kwargs)

                fetched_records_2: List[BaseDTO] = store_2.list(*opt, **kwargs)
                count_2 = store_2.count(**kwargs)

                assert count_1 == count_2
                assert len(fetched_records_1) == len(fetched_records_2)

                for i in range(len(fetched_records_1)):
                    assert fetched_records_1[i].primary_id == fetched_records_2[i].primary_id
       