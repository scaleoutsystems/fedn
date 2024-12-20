import graphene

from fedn.network.api.v1.shared import session_store


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

    def resolve_session_config(self, info):
        return self["session_config"]


class Query(graphene.ObjectType):
    session = graphene.List(
        SessionType,
        id=graphene.String(),
    )

    def resolve_session(root, info, id=None):
        result = None
        if id:
            response = session_store.get(id)
            result = []
            result.append(response)
        else:
            response = session_store.list(0, 0, None)
            result = response["result"]

        return result


schema = graphene.Schema(query=Query)
