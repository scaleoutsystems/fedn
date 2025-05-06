from typing import Dict, List

from pymongo.database import Database

from fedn.network.storage.statestore.stores.dto.attribute import AttributeDTO
from fedn.network.storage.statestore.stores.shared import SortOrder
from fedn.network.storage.statestore.stores.sql.shared import AttributeModel, from_orm_model
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store, from_document


class AttributeStore(Store[AttributeDTO]):
    def get_current_attributes_for_client(self, client_id: str) -> List[AttributeDTO]:
        """Get all attributes for a specific client.

        This method returns the most recent attributes for the given client_id.
        """
        attributes = self.list(limit=0, skip=0, sort_key="committed_at", sort_order=SortOrder.DESCENDING, **{"sender.client_id": client_id})
        keys = {attribute.key for attribute in attributes}

        result = []
        for key in keys:
            result.append([attribute for attribute in attributes if attribute.key == key][0])
        return result


class MongoDBAttributeStore(AttributeStore, MongoDBStore[AttributeDTO]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection, "attribute_id")

    def _document_from_dto(self, item: AttributeDTO) -> Dict:
        item_dict = item.to_db(exclude_unset=False)
        return item_dict

    def _dto_from_document(self, document: Dict) -> AttributeDTO:
        item = from_document(document)
        return AttributeDTO().patch_with(item, throw_on_extra_keys=False)


def _translate_key_sql(key: str):
    if key == "sender.name":
        key = "sender_name"
    elif key == "sender.role":
        key = "sender_role"
    elif key == "sender.client_id":
        key = "sender_client_id"
    return key


class SQLAttributeStore(AttributeStore, SQLStore[AttributeDTO, AttributeModel]):
    def __init__(self, session):
        super().__init__(session, AttributeModel, "attribute_id")

    def list(self, limit=0, skip=0, sort_key=None, sort_order=SortOrder.DESCENDING, **kwargs):
        sort_key = _translate_key_sql(sort_key)
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        return super().list(limit, skip, sort_key, sort_order, **kwargs)

    def count(self, **kwargs):
        kwargs = {_translate_key_sql(k): v for k, v in kwargs.items()}
        return super().count(**kwargs)

    def _update_orm_model_from_dto(self, entity: AttributeModel, item: AttributeDTO):
        item_dict = item.to_db(exclude_unset=False)
        item_dict["id"] = item_dict.pop("attribute_id", None)

        sender: Dict = item_dict.pop("sender", {})
        item_dict["sender_name"] = sender.get("name")
        item_dict["sender_role"] = sender.get("role")
        item_dict["sender_client_id"] = sender.get("client_id")

        for key, value in item_dict.items():
            setattr(entity, key, value)
        return entity

    def _dto_from_orm_model(self, item: AttributeModel) -> AttributeDTO:
        orm_dict = from_orm_model(item, AttributeModel)
        orm_dict["attribute_id"] = orm_dict.pop("id")
        orm_dict["sender"] = {
            "name": orm_dict.pop("sender_name"),
            "role": orm_dict.pop("sender_role"),
            "client_id": orm_dict.pop("sender_client_id"),
        }
        return AttributeDTO().populate_with(orm_dict)
