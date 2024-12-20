import graphene
import pymongo

from fedn.network.api.v1.shared import model_store, session_store, validation_store


class ActorType(graphene.ObjectType):
    name = graphene.String()
    role = graphene.String()


class ValidationType(graphene.ObjectType):
    correlationId = graphene.String()  # noqa: N815
    data = graphene.String()
    id = graphene.String()
    meta = graphene.String()
    modelId = graphene.String()  # noqa: N815
    receiver = graphene.Field(ActorType)
    sender = graphene.Field(ActorType)
    sessionId = graphene.String()  # noqa: N815
    timestamp = graphene.String()

    def resolve_receiver(self, info):
        return self["receiver"]

    def resolve_sender(self, info):
        return self["sender"]


class ModelType(graphene.ObjectType):
    id = graphene.String()
    model = graphene.String()
    name = graphene.String()
    committed_at = graphene.DateTime()
    session_id = graphene.String()
    parent_model = graphene.String()
    validation = graphene.List(ValidationType)

    def resolve_validation(self, info):
        kwargs = {"modelId": self["model"]}
        response = validation_store.list(0, 0, None, sort_order=pymongo.DESCENDING, use_typing=False, **kwargs)
        result = response["result"]

        return result


class SessionConfigType(graphene.ObjectType):
    aggregator = graphene.String()
    round_timeout = graphene.Int()
    buffer_size = graphene.Int()
    model_id = graphene.String()
    delete_models_storage = graphene.Boolean()
    clients_required = graphene.Int()
    helper_type = graphene.String()
    validate = graphene.Boolean()
    session_id = graphene.String()
    model_id = graphene.String()


class SessionType(graphene.ObjectType):
    id = graphene.String()
    session_id = graphene.String()
    name = graphene.String()
    committed_at = graphene.DateTime()
    session_config = graphene.Field(SessionConfigType)
    model = graphene.List(ModelType)
    validation = graphene.List(ValidationType)

    def resolve_session_config(self, info):
        return self["session_config"]

    def resolve_model(self, info):
        kwargs = {"session_id": self["session_id"]}
        response = model_store.list(0, 0, None, sort_order=pymongo.DESCENDING, use_typing=False, **kwargs)
        result = response["result"]

        return result

    def resolve_validation(self, info):
        kwargs = {"sessionId": self["session_id"]}
        response = validation_store.list(0, 0, None, sort_order=pymongo.DESCENDING, use_typing=False, **kwargs)
        result = response["result"]

        return result


class Query(graphene.ObjectType):
    session = graphene.List(
        SessionType,
        id=graphene.String(),
        name=graphene.String(),
    )

    model = graphene.List(
        ModelType,
        id=graphene.String(),
        session_id=graphene.String(),
    )

    validation = graphene.List(
        ValidationType,
        id=graphene.String(),
        session_id=graphene.String(),
    )

    def resolve_session(root, info, id: str = None, name: str = None):
        result = None
        if id:
            response = session_store.get(id)
            result = []
            result.append(response)
        elif name:
            kwargs = {"name": name}
            response = session_store.list(0, 0, None, sort_order=pymongo.DESCENDING, use_typing=False, **kwargs)
            result = response["result"]
        else:
            response = session_store.list(0, 0, None)
            result = response["result"]

        return result

    def resolve_model(root, info, id: str = None, session_id: str = None):
        result = None
        if id:
            response = model_store.get(id)
            result = []
            result.append(response)
        elif session_id:
            kwargs = {"session_id": session_id}
            response = model_store.list(0, 0, None, sort_order=pymongo.DESCENDING, use_typing=False, **kwargs)
            result = response["result"]
        else:
            response = model_store.list(0, 0, None)
            result = response["result"]

        return result

    def resolve_validation(root, info, id: str = None, session_id: str = None):
        result = None
        if id:
            response = validation_store.get(id)
            result = []
            result.append(response)
        elif session_id:
            kwargs = {"session_id": session_id}
            response = validation_store.list(0, 0, None, sort_order=pymongo.DESCENDING, use_typing=False, **kwargs)
            result = response["result"]
        else:
            response = validation_store.list(0, 0, None)
            result = response["result"]

        return result


schema = graphene.Schema(query=Query)
