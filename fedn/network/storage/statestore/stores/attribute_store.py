from abc import abstractmethod
from typing import Dict

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.attribute import AttributeDTO
from fedn.network.storage.statestore.stores.sql.shared import AttributeModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class AttributeStore(Store[AttributeDTO]):
    @abstractmethod
    def get_by_client_id(self, client_id: str) -> AttributeDTO:
        """Get the attribute by client ID. If multiple attributes are found, return the most recent.

        Args:
            client_id (str): The client ID.

        Returns:
            AttributeDTO: The attribute data transfer object.

        """
        res = self.list(1, 0, sort_key="committed_at", sort_order=pymongo.DESCENDING, sender_client_id=client_id)
        if res:
            return res[0]
        else:
            return None


def _translate_key_sql(key: str):
    if key == "sender.name":
        key = "sender_name"
    elif key == "sender.role":
        key = "sender_role"
    return key


class MongoDBMetricStore(AttributeStore, MongoDBStore[AttributeDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "attribute_id")

    def _document_from_dto(self, item: AttributeDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> AttributeDTO:
        item = from_document(document)
        return AttributeDTO().patch_with(item, throw_on_extra_keys=False)


class SQLMetricStore(AttributeStore, SQLStore[AttributeDTO, AttributeModel]):
    def __init__(self, session):
        super().__init__(session, AttributeModel)

    def list(self, limit=0, skip=0, sort_key=None, sort_order=pymongo.DESCENDING, **kwargs):
        sort_key = _translate_key_sql(sort_key)
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        return super().list(limit, skip, sort_key, sort_order, **kwargs)

    def count(self, **kwargs):
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        return super().count(**kwargs)

    def _update_orm_model_from_dto(self, entity: AttributeModel, item: AttributeDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("attribute_id", None)
        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: AttributeModel) -> AttributeDTO:
        orm_dict = from_orm_model(item, AttributeModel)
        orm_dict["attribute_id"] = orm_dict.pop("id")
        return AttributeDTO().populate_with(orm_dict)
