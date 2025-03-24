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
        result = validation_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        result = [validation.to_dict() for validation in result]
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
        all_models = model_store.list(**kwargs)
        result = [model.to_dict() for model in all_models]

        return result

    def resolve_validations(self, info):
        kwargs = {"sessionId": self["session_id"]}
        result = validation_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        result = [validation.to_dict() for validation in result]

        return result

    def resolve_statuses(self, info):
        kwargs = {"sessionId": self["session_id"]}
        result = status_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        result = [status.to_dict() for status in result]

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

        return result.to_dict()

    def resolve_sessions(root, info, name: str = None):
        if name:
            kwargs = {"name": name}
        else:
            kwargs = {}
        all_models = model_store.list(**kwargs)
        result = [model.to_dict() for model in all_models]
        return result

    def resolve_model(root, info, id: str = None):
        result = model_store.get(id).to_dict()

        return result

    def resolve_models(root, info, session_id: str = None):
        if session_id:
            kwargs = {"session_id": session_id}
        else:
            kwargs = {}
        all_models = model_store.list(**kwargs)
        result = [model.to_dict() for model in all_models]
        return result

    def resolve_validation(root, info, id: str = None):
        result = validation_store.get(id).to_dict()

        return result

    def resolve_validations(root, info, session_id: str = None):
        if session_id:
            kwargs = {"session_id": session_id}
            result = validation_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        else:
            result = validation_store.list(0, 0, None)

        return [validation.to_dict() for validation in result]

    def resolve_status(root, info, id: str = None):
        result = status_store.get(id).to_dict()

        return result

    def resolve_statuses(root, info, session_id: str = None):
        if session_id:
            kwargs = {"sessionId": session_id}
            result = status_store.list(0, 0, None, sort_order=pymongo.DESCENDING, **kwargs)
        else:
            result = status_store.list(0, 0, None)

        return [status.to_dict() for status in result]


schema = graphene.Schema(query=Query)
