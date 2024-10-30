from typing import Any, Dict, List, Tuple

import pymongo
from pymongo.database import Database

from fedn.network.storage.statestore.stores.store import Store


class Prediction:
    def __init__(
        self, id: str, model_id: str, data: str, correlation_id: str, timestamp: str, prediction_id: str, meta: str, sender: dict = None, receiver: dict = None
    ):
        self.id = id
        self.model_id = model_id
        self.data = data
        self.correlation_id = correlation_id
        self.timestamp = timestamp
        self.prediction_id = prediction_id
        self.meta = meta
        self.sender = sender
        self.receiver = receiver

    def from_dict(data: dict) -> "Prediction":
        return Prediction(
            id=str(data["_id"]),
            model_id=data["modelId"] if "modelId" in data else None,
            data=data["data"] if "data" in data else None,
            correlation_id=data["correlationId"] if "correlationId" in data else None,
            timestamp=data["timestamp"] if "timestamp" in data else None,
            prediction_id=data["predictionId"] if "predictionId" in data else None,
            meta=data["meta"] if "meta" in data else None,
            sender=data["sender"] if "sender" in data else None,
            receiver=data["receiver"] if "receiver" in data else None,
        )


class PredictionStore(Store[Prediction]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)

    def get(self, id: str, use_typing: bool = False) -> Prediction:
        """Get an entity by id
        param id: The id of the entity
            type: str
            description: The id of the entity, can be either the id or the Prediction (property)
        param use_typing: Whether to return the entity as a typed object or as a dict
            type: bool
        return: The entity
        """
        response = super().get(id, use_typing=use_typing)
        return Prediction.from_dict(response) if use_typing else response

    def update(self, id: str, item: Prediction) -> bool:
        raise NotImplementedError("Update not implemented for PredictionStore")

    def add(self, item: Prediction) -> Tuple[bool, Any]:
        return super().add(item)

    def delete(self, id: str) -> bool:
        raise NotImplementedError("Delete not implemented for PredictionStore")

    def list(self, limit: int, skip: int, sort_key: str, sort_order=pymongo.DESCENDING, use_typing: bool = False, **kwargs) -> Dict[int, List[Prediction]]:
        """List entities
        param limit: The maximum number of entities to return
            type: int
            description: The maximum number of entities to return
        param skip: The number of entities to skip
            type: int
            description: The number of entities to skip
        param sort_key: The key to sort by
            type: str
            description: The key to sort by
        param sort_order: The order to sort by
            type: pymongo.DESCENDING
            description: The order to sort by
        param use_typing: Whether to return the entities as typed objects or as dicts
            type: bool
            description: Whether to return the entities as typed objects or as dicts
        return: A dictionary with the count and a list of entities
        """
        response = super().list(limit, skip, sort_key or "timestamp", sort_order, use_typing=use_typing, **kwargs)

        result = [Prediction.from_dict(item) for item in response["result"]] if use_typing else response["result"]
        return {"count": response["count"], "result": result}
