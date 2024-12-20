import graphene
import pymongo

from fedn.network.api.v1.shared import model_store, session_store


class ModelType(graphene.ObjectType):
    id = graphene.String()
    model = graphene.String()
    name = graphene.String()
    committed_at = graphene.DateTime()
    session_id = graphene.String()
    parent_model = graphene.String()


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

    def resolve_session_config(self, info):
        return self["session_config"]

    def resolve_model(self, info):
        kwargs = {"session_id": self["session_id"]}
        response = model_store.list(0, 0, None, sort_order=pymongo.DESCENDING, use_typing=False, **kwargs)
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
    )

    def resolve_session(root, info, id=None, name=None):
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

    def resolve_model(root, info, id=None):
        result = None
        if id:
            response = model_store.get(id)
            result = []
            result.append(response)
        else:
            response = model_store.list(0, 0, None)
            result = response["result"]

        return result


schema = graphene.Schema(query=Query)
