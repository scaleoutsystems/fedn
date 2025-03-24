import graphene
import pymongo

from fedn.network.api.shared import model_store, session_store, status_store, validation_store


class ActorType(graphene.ObjectType):
    name = graphene.String()
    role = graphene.String()


class StatusType(graphene.ObjectType):
    data = graphene.String()
    extra = graphene.String()
    id = graphene.String()
    logLevel = graphene.String()  # noqa: N815
    sender = graphene.Field(ActorType)
    sessionId = graphene.String()  # noqa: N815
    status = graphene.String()
    timestamp = graphene.String()
    type = graphene.String()


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
    validations = graphene.List(ValidationType)

    def resolve_validations(self, info):
        kwargs = {"modelId": self["model"]}
        response = validation_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
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
    models = graphene.List(ModelType)
    validations = graphene.List(ValidationType)
    statuses = graphene.List(StatusType)

    def resolve_session_config(self, info):
        return self["session_config"]

    def resolve_models(self, info):
        kwargs = {"session_id": self["session_id"]}
        response = model_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        result = response["result"]

        return result

    def resolve_validations(self, info):
        kwargs = {"sessionId": self["session_id"]}
        response = validation_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        result = response["result"]

        return result

    def resolve_statuses(self, info):
        kwargs = {"sessionId": self["session_id"]}
        response = status_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        result = response["result"]

        return result


class Query(graphene.ObjectType):
    session = graphene.Field(
        SessionType,
        id=graphene.String(required=True),
    )
    sessions = graphene.List(
        SessionType,
        name=graphene.String(),
    )

    model = graphene.Field(
        ModelType,
        id=graphene.String(required=True),
    )

    models = graphene.List(
        ModelType,
        session_id=graphene.String(),
    )

    validation = graphene.Field(
        ValidationType,
        id=graphene.String(required=True),
    )

    validations = graphene.List(
        ValidationType,
        session_id=graphene.String(),
    )

    status = graphene.Field(
        StatusType,
        id=graphene.String(required=True),
    )

    statuses = graphene.List(
        StatusType,
        session_id=graphene.String(),
    )

    def resolve_session(root, info, id: str = None):
        result = session_store.get(id)

        return result

    def resolve_sessions(root, info, name: str = None):
        response = None
        if name:
            kwargs = {"name": name}
            response = session_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        else:
            response = session_store.list(0, 0, None)

        return response["result"]

    def resolve_model(root, info, id: str = None):
        result = model_store.get(id)

        return result

    def resolve_models(root, info, session_id: str = None):
        response = None
        if session_id:
            kwargs = {"session_id": session_id}
            response = model_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        else:
            response = model_store.list(0, 0, None)

        return response["result"]

    def resolve_validation(root, info, id: str = None):
        result = validation_store.get(id)

        return result

    def resolve_validations(root, info, session_id: str = None):
        response = None
        if session_id:
            kwargs = {"session_id": session_id}
            response = validation_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        else:
            response = validation_store.list(0, 0, None)

        return response["result"]

    def resolve_status(root, info, id: str = None):
        result = status_store.get(id)

        return result

    def resolve_statuses(root, info, session_id: str = None):
        response = None
        if session_id:
            kwargs = {"sessionId": session_id}
            response = status_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        else:
            response = status_store.list(0, 0, None)

        return response["result"]


schema = graphene.Schema(query=Query)
