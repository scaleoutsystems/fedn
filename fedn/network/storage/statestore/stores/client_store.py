from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import pymongo
from bson import ObjectId
from pymongo.database import Database
from sqlalchemy import func, or_, select

from fedn.network.storage.statestore.models import Client
from fedn.network.storage.statestore.stores.sql.shared import ClientModel
from fedn.network.storage.statestore.stores.store import MongoDBStore, SQLStore, Store

from .shared import from_document


class ClientStore(Store[Client]):
    @abstractmethod
    def upsert(self, item: Client) -> Tuple[bool, Any]:
        pass

    @abstractmethod
    def connected_client_count(self, combiners: List[str]) -> List[Client]:
        pass

    @abstractmethod
    def commit(self, item: Client) -> Tuple[bool, Any]:
        pass


class MongoDBClientStore(ClientStore, MongoDBStore[Client]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        # TODO: Could there be side effects from using unique=True here?
        self.database[self.collection].create_index([("client_id", pymongo.DESCENDING)], unique=True)

    def get(self, id: str) -> Client:
        """Get an entity by id
        param id: The id of the entity
            type: str
        return: The entity
        """
        if ObjectId.is_valid(id):
            response = super().get(id)
        else:
            response = self._get_client_by_client_id(id)
        return Client(**response) if response is not None else None

    def _get_client_by_client_id(self, client_id: str) -> Dict:
        document = self.database[self.collection].find_one({"client_id": client_id})
        if document is None:
            return None
        return from_document(document)

    def update(self, id: str, item: Dict) -> Tuple[bool, Any]:
        try:
            existing_client = self.get(id)

            return super().update(existing_client.id, item)
        except Exception as e:
            return False, str(e)

    def add(self, item: Client) -> Tuple[bool, Any]:
        """Add an entity
        param item: The entity to add
            type: Client
        return: A tuple with a boolean and a message if failure or the added entity if success
        """
        item_dict = item.to_dict()
        if "id" in item_dict:
            del item_dict["id"]
        success, obj = super().add(item_dict)
        if success:
            obj = Client(**obj)
        return success, obj

    def upsert(self, item: Client) -> Tuple[bool, Any]:
        """Update an existing entity or add a new entity
        param item: The entity to add or update
            type: Client
        return: A tuple with a boolean and a message if failure or the added or updated entity if success
        """
        try:
            item_dict = item.to_dict()
            if "id" in item_dict:
                del item_dict["id"]
            result = self.database[self.collection].update_one({"client_id": item.client_id}, {"$set": item_dict}, upsert=True)
            id = result.upserted_id
            document = self.database[self.collection].find_one({"_id": id})
            return True, Client(**from_document(document))
        except Exception as e:
            return False, str(e)

    # TODO: This method has the same funcitonality as update but different signature. Should we change the signature of update and replace update with this?
    def commit(self, item: Client) -> Tuple[bool, Any]:
        """Update an existing entity
        param item: The entity to update
            type: Client
        return: A tuple with a boolean and a message if failure or the updated entity if success
        """
        try:
            if ObjectId.is_valid(item.id):
                document_id = item.id
            else:
                document_id = self._get_client_by_client_id["id"]
            item_dict = item.to_dict()
            if "id" in item_dict:
                del item_dict["id"]
            success, obj = super().update(document_id, item_dict)
            if success:
                obj = Client(**obj)
            return success, obj
        except Exception as e:
            return False, str(e)

    def delete(self, id: str) -> bool:
        """Delete an entity
        param id: The id of the entity
            type: str
        return: A boolean indicating success or failure
        """
        kwargs = {"_id": ObjectId(id)} if ObjectId.is_valid(id) else {"client_id": id}

        document = self.database[self.collection].find_one(kwargs)

        if document is None:
            return False

        return super().delete(document["_id"])

    def select(self, **filter_kwargs) -> List[Client]:
        """Select entities with a filter
        param filter_kwargs: The filter parameters
        return: A list of entities
        """
        results = super().list(0, 0, "last_seen", pymongo.DESCENDING, **filter_kwargs)["result"]
        return [Client(**result) for result in results]

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs) -> Dict[int, List[Dict]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
        param skip: The number of entities to skip
            type: int
        param sort_key: The key to sort by
            type: str
        param sort_order: The order to sort by
            type: pymongo.DESCENDING | pymongo.ASCENDING
        param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
        return: A dictionary with the count and the result
        """
        return super().list(limit, skip, sort_key or "last_seen", sort_order, **kwargs)

    def count(self, **kwargs) -> int:
        """Count entities
        param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
        return: The count of entities
        """
        return super().count(**kwargs)

    def connected_client_count(self, combiners: List[str]) -> List:
        """Count the number of connected clients for each combiner.

        :param combiners: list of combiners to get data for.
        :type combiners: list
        :param sort_key: The key to sort by.
        :type sort_key: str
        :param sort_order: The sort order.
        :type sort_order: pymongo.ASCENDING or pymongo.DESCENDING
        :return: list of combiner data.
        :rtype: list(ObjectId)
        """
        try:
            pipeline = (
                [
                    {"$match": {"combiner": {"$in": combiners}, "status": "online"}},
                    {"$group": {"_id": "$combiner", "count": {"$sum": 1}}},
                    {"$project": {"id": "$_id", "count": 1, "_id": 0}},
                ]
                if len(combiners) > 0
                else [
                    {"$match": {"status": "online"}},
                    {"$group": {"_id": "$combiner", "count": {"$sum": 1}}},
                    {"$project": {"id": "$_id", "count": 1, "_id": 0}},
                ]
            )

            result = list(self.database[self.collection].aggregate(pipeline))
        except Exception:
            result = []

        return result


def row_to_client(row: ClientModel) -> Client:
    return Client(
        id=row.id,
        client_id=row.client_id,
        combiner=row.combiner,
        combiner_preferred=row.combiner_preferred,
        ip=row.ip,
        name=row.name,
        package=row.package,
        status=row.status,
        last_seen=row.last_seen,
    )


class SQLClientStore(ClientStore, SQLStore[Client]):
    def __init__(self, Session):
        super().__init__(Session)

    def get(self, id: str) -> Client:
        """Get an entity by id
        param id: The id of the entity
            type: str
        return: The entity
        """
        with self.Session() as session:
            stmt = select(ClientModel).where(or_(ClientModel.id == id, ClientModel.client_id == id))
            item = session.scalars(stmt).first()

            if item is None:
                return None

            return row_to_client(item)

    def update(self, id: str, item: Dict) -> Tuple[bool, Any]:
        with self.Session() as session:
            stmt = select(ClientModel).where(or_(ClientModel.id == id, ClientModel.client_id == id))
            existing_item = session.scalars(stmt).first()

            if existing_item is None:
                return False, "Item not found"

            existing_item.combiner = item.get("combiner")
            existing_item.combiner_preferred = item.get("combiner_preferred")
            existing_item.ip = item.get("ip")
            existing_item.name = item.get("name")
            existing_item.package = item.get("package")
            existing_item.status = item.get("status")
            existing_item.last_seen = item.get("last_seen")

            session.commit()

            return True, row_to_client(existing_item)

    def add(self, item: Client) -> Tuple[bool, Any]:
        """Add an entity
        param item: The entity to add
            type: Client
        return: A tuple with a boolean and a message if failure or the added entity if success
        """
        with self.Session() as session:
            entity = ClientModel(
                client_id=item.client_id,
                combiner=item.combiner,
                combiner_preferred=item.combiner_preferred,
                ip=item.ip,
                name=item.name,
                package=item.package,
                status=item.status,
                last_seen=item.last_seen,
            )

            session.add(entity)
            session.commit()

            return True, row_to_client(entity)

    def upsert(self, item: Client) -> Tuple[bool, Any]:
        with self.Session() as session:
            client_id = item.get("client_id")

            stmt = select(ClientModel).where(ClientModel.client_id == client_id)
            existing_item = session.scalars(stmt).first()

            if existing_item is None:
                return self.add(item)

            return self.commit(item)

    def delete(self, id) -> bool:
        """Delete an entity
        param id: The id of the entity
            type: str
        return: A boolean indicating success or failure
        """
        with self.Session() as session:
            stmt = select(ClientModel).where(or_(ClientModel.id == id, ClientModel.client_id == id))
            item = session.scalars(stmt).first()

            if item is None:
                return False

            session.delete(item)
            session.commit()

            return True

    # TODO: This method has the same funcitonality as update but different signature. Should we change the signature of update and replace update with this?
    def commit(self, item: Client) -> Tuple[bool, Any]:
        """Update an existing entity
        param item: The entity to update
            type: Client
        return: A tuple with a boolean and a message if failure or the updated entity if success
        """
        with self.Session() as session:
            stmt = select(ClientModel).where(ClientModel.client_id == item.client_id)
            existing_item = session.scalars(stmt).first()

            if existing_item is None:
                return False, "Item not found"
            existing_item.combiner = item.combiner
            existing_item.combiner_preferred = item.combiner_preferred
            existing_item.ip = item.ip
            existing_item.name = item.name
            existing_item.package = item.package
            existing_item.status = item.status
            existing_item.last_seen = item.last_seen

            session.commit()

            return True, row_to_client(existing_item)

    def select(self, **filter_kwargs) -> List[Client]:
        """Select entities with a filter
        param filter_kwargs: The filter parameters
        return: A list of entities
        """
        results = self.list(0, 0, "committed_at", pymongo.DESCENDING, **filter_kwargs)["result"]
        return [row_to_client(result) for result in results]

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, **kwargs):
        """List entities
        param limit: The maximum number of entities to return
            type: int
            param skip: The number of entities to skip
            type: int
            param sort_key: The key to sort by
            type: str
            param sort_order: The order to sort by
            type: pymongo.DESCENDING | pymongo.ASCENDING
            param kwargs: Additional query parameters
            type: dict
            example: {"key": "models"}
            return: A dictionary with the count and the result
        """
        with self.Session() as session:
            stmt = select(ClientModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(ClientModel, key) == value)

            _sort_order: str = "DESC" if sort_order == pymongo.DESCENDING else "ASC"
            _sort_key: str = sort_key or "committed_at"

            if _sort_key in ClientModel.__table__.columns:
                sort_obj = ClientModel.__table__.columns.get(_sort_key) if _sort_order == "ASC" else ClientModel.__table__.columns.get(_sort_key).desc()

                stmt = stmt.order_by(sort_obj)

            if limit:
                stmt = stmt.offset(skip or 0).limit(limit)
            elif skip:
                stmt = stmt.offset(skip)

            items = session.scalars(stmt).all()

            result = []
            for item in items:
                result.append(row_to_client(item).to_dict())

            count = session.scalar(select(func.count()).select_from(ClientModel))

            return {"count": count, "result": result}

    def count(self, **kwargs):
        with self.Session() as session:
            stmt = select(func.count()).select_from(ClientModel)

            for key, value in kwargs.items():
                stmt = stmt.where(getattr(ClientModel, key) == value)

            count = session.scalar(stmt)

            return count

    def connected_client_count(self, combiners) -> List[Dict]:
        with self.Session() as session:
            stmt = select(ClientModel.combiner, func.count(ClientModel.combiner)).group_by(ClientModel.combiner)
            if combiners:
                stmt = stmt.where(ClientModel.combiner.in_(combiners))

            items = session.execute(stmt).fetchall()

            result = []
            for item in items:
                result.append({"combiner": item[0], "count": item[1]})

            return result
