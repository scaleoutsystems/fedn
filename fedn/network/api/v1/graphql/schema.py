import graphene

from fedn.network.controller.control import Control
from fedn.network.storage.statestore.stores.shared import SortOrder


def get_sort_order_from_string(sort_order: str) -> SortOrder:
    """Convert a string to a SortOrder enum."""
    if sort_order.lower() == "asc":
        return SortOrder.ASCENDING
    elif sort_order.lower() == "desc":
        return SortOrder.DESCENDING
    else:
        raise ValueError(f"Invalid sort order: {sort_order}")


class ActorType(graphene.ObjectType):
    name = graphene.String()
    role = graphene.String()


class StatusType(graphene.ObjectType):
    data = graphene.String()
    extra = graphene.String()
    logLevel = graphene.String()  # noqa: N815
    sender = graphene.Field(ActorType)
    session_id = graphene.String()  # noqa: N815
    status = graphene.String()
    timestamp = graphene.String()
    type = graphene.String()
    session = graphene.Field(lambda: SessionType)

    def resolve_session(self, info):
        db = Control.instance().db
        session = db.session_store.get(self["session_id"])
        if session:
            return session.to_dict()
        return None


class ValidationType(graphene.ObjectType):
    correlation_id = graphene.String()  # noqa: N815
    data = graphene.String()
    meta = graphene.String()
    model_id = graphene.String()  # noqa: N815
    receiver = graphene.Field(ActorType)
    sender = graphene.Field(ActorType)
    session_id = graphene.String()  # noqa: N815
    timestamp = graphene.String()
    session = graphene.Field(lambda: SessionType)

    def resolve_receiver(self, info):
        return self["receiver"]

    def resolve_sender(self, info):
        return self["sender"]

    def resolve_session(self, info):
        db = Control.instance().db
        session = db.session_store.get(self["session_id"])
        if session:
            return session.to_dict()
        return None


class ModelType(graphene.ObjectType):
    name = graphene.String()
    committed_at = graphene.DateTime()
    model_id = graphene.String()
    session_id = graphene.String()
    parent_model = graphene.String()
    validations = graphene.List(
        ValidationType,
        limit=graphene.Int(required=False, default_value=0),
        skip=graphene.Int(required=False, default_value=0),
        sort_key=graphene.String(required=False, default_value="committed_at"),
        sort_order=graphene.String(required=False, default_value="desc"),
    )
    session = graphene.Field(lambda: SessionType)

    def resolve_validations(self, info, limit=0, skip=0, sort_key="committed_at", sort_order="desc"):
        db = Control.instance().db
        kwargs = {"model_id": self["model_id"]}
        sort_order = get_sort_order_from_string(sort_order)

        validations = db.validation_store.list(limit=limit, skip=skip, sort_key=sort_key, sort_order=sort_order, **kwargs)
        result = [validation.to_dict() for validation in validations]
        return result

    def resolve_session(self, info):
        db = Control.instance().db
        session = db.session_store.get(self["session_id"])
        if session:
            return session.to_dict()
        return None


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
    session_id = graphene.String()
    name = graphene.String()
    committed_at = graphene.DateTime()
    session_config = graphene.Field(SessionConfigType)
    models = graphene.List(
        ModelType,
        limit=graphene.Int(required=False, default_value=0),
        skip=graphene.Int(required=False, default_value=0),
        sort_key=graphene.String(required=False, default_value="committed_at"),
        sort_order=graphene.String(required=False, default_value="desc"),
    )
    validations = graphene.List(
        ValidationType,
        limit=graphene.Int(required=False, default_value=0),
        skip=graphene.Int(required=False, default_value=0),
        sort_key=graphene.String(required=False, default_value="committed_at"),
        sort_order=graphene.String(required=False, default_value="desc"),
    )
    statuses = graphene.List(
        StatusType,
        limit=graphene.Int(required=False, default_value=0),
        skip=graphene.Int(required=False, default_value=0),
        sort_key=graphene.String(required=False, default_value="committed_at"),
        sort_order=graphene.String(required=False, default_value="desc"),
    )

    def resolve_session_config(self, info):
        return self["session_config"]

    def resolve_models(self, info, limit=0, skip=0, sort_key="committed_at", sort_order="desc"):
        db = Control.instance().db

        kwargs = {"session_id": self["session_id"]}
        sort_order = get_sort_order_from_string(sort_order)

        models = db.model_store.list(limit=limit, skip=skip, sort_key=sort_key, sort_order=sort_order, **kwargs)
        result = [model.to_dict() for model in models]

        return result

    def resolve_validations(self, info, limit=0, skip=0, sort_key="committed_at", sort_order="desc"):
        db = Control.instance().db

        kwargs = {"session_id": self["session_id"]}
        sort_order = get_sort_order_from_string(sort_order)

        validations = db.validation_store.list(limit=limit, skip=skip, sort_key=sort_key, sort_order=sort_order, **kwargs)
        result = [validation.to_dict() for validation in validations]

        return result

    def resolve_statuses(self, info, limit=0, skip=0, sort_key="committed_at", sort_order="desc"):
        db = Control.instance().db

        kwargs = {"session_id": self["session_id"]}
        sort_order = get_sort_order_from_string(sort_order)

        statuses = db.status_store.list(limit=limit, skip=skip, sort_key=sort_key, sort_order=sort_order, **kwargs)
        result = [status.to_dict() for status in statuses]

        return result


class Query(graphene.ObjectType):
    session = graphene.Field(
        SessionType,
        id=graphene.String(required=True),
    )
    sessions = graphene.List(
        SessionType,
        name=graphene.String(),
        limit=graphene.Int(required=False, default_value=25),
        skip=graphene.Int(required=False, default_value=0),
        sort_key=graphene.String(required=False, default_value="committed_at"),
        sort_order=graphene.String(required=False, default_value="desc"),
    )

    model = graphene.Field(
        ModelType,
        id=graphene.String(required=True),
    )

    models = graphene.List(
        ModelType,
        session_id=graphene.String(),
        limit=graphene.Int(required=False, default_value=25),
        skip=graphene.Int(required=False, default_value=0),
        sort_key=graphene.String(required=False, default_value="committed_at"),
        sort_order=graphene.String(required=False, default_value="desc"),
    )

    validation = graphene.Field(
        ValidationType,
        id=graphene.String(required=True),
    )

    validations = graphene.List(
        ValidationType,
        session_id=graphene.String(),
        limit=graphene.Int(required=False, default_value=25),
        skip=graphene.Int(required=False, default_value=0),
        sort_key=graphene.String(required=False, default_value="committed_at"),
        sort_order=graphene.String(required=False, default_value="desc"),
    )

    status = graphene.Field(
        StatusType,
        id=graphene.String(required=True),
    )

    statuses = graphene.List(
        StatusType,
        session_id=graphene.String(),
        limit=graphene.Int(required=False, default_value=25),
        skip=graphene.Int(required=False, default_value=0),
        sort_key=graphene.String(required=False, default_value="committed_at"),
        sort_order=graphene.String(required=False, default_value="desc"),
    )

    def resolve_session(root, info, id: str = None):
        db = Control.instance().db
        result = db.session_store.get(id)

        return result.to_dict()

    def resolve_sessions(root, info, name: str = None, limit: int = 25, skip: int = 0, sort_key: str = "committed_at", sort_order: str = "desc"):
        db = Control.instance().db
        if name:
            kwargs = {"name": name}
        else:
            kwargs = {}

        sort_order = get_sort_order_from_string(sort_order)

        sessions = db.session_store.list(limit=limit, skip=skip, sort_key=sort_key, sort_order=sort_order, **kwargs)
        result = [session.to_dict() for session in sessions]

        return result

    def resolve_model(root, info, id: str = None):
        db = Control.instance().db
        result = db.model_store.get(id).to_dict()

        return result

    def resolve_models(root, info, session_id: str = None, limit: int = 25, skip: int = 0, sort_key: str = "committed_at", sort_order: str = "desc"):
        db = Control.instance().db
        if session_id:
            kwargs = {"session_id": session_id}
        else:
            kwargs = {}

        sort_order = get_sort_order_from_string(sort_order)

        models = db.model_store.list(limit=limit, skip=skip, sort_key=sort_key, sort_order=sort_order, **kwargs)
        result = [model.to_dict() for model in models]

        return result

    def resolve_validation(root, info, id: str = None):
        db = Control.instance().db
        result = db.validation_store.get(id).to_dict()

        return result

    def resolve_validations(root, info, session_id: str = None, limit: int = 25, skip: int = 0, sort_key: str = "committed_at", sort_order: str = "desc"):
        db = Control.instance().db

        if session_id:
            kwargs = {"session_id": session_id}
        else:
            kwargs = {}

        sort_order = get_sort_order_from_string(sort_order)

        validations = db.validation_store.list(limit=limit, skip=skip, sort_key=sort_key, sort_order=sort_order, **kwargs)
        result = [validation.to_dict() for validation in validations]

        return result

    def resolve_status(root, info, id: str = None):
        db = Control.instance().db
        result = db.status_store.get(id).to_dict()

        return result

    def resolve_statuses(root, info, session_id: str = None, limit: int = 25, skip: int = 0, sort_key: str = "committed_at", sort_order: str = "desc"):
        db = Control.instance().db

        if session_id:
            kwargs = {"session_id": session_id}
        else:
            kwargs = {}

        sort_order = get_sort_order_from_string(sort_order)

        statuses = db.status_store.list(limit=limit, skip=skip, sort_key=sort_key, sort_order=sort_order, **kwargs)
        result = [status.to_dict() for status in statuses]

        return result


schema = graphene.Schema(query=Query)
