import graphene


class SessionType(graphene.ObjectType):
    id = graphene.String()
    name = graphene.String()


class Query(graphene.ObjectType):
    session = graphene.List(
        SessionType,
        id=graphene.Int(),
        name=graphene.String(),
    )

    def resolve_hello(root, info):
        return "Hello, GraphQL!"

    def resolve_session(root, info, id=None, name=None, pet_name=None):
        return [
            {"id": "1", "name": "Session 1"},
            {"id": "2", "name": "Session 2"},
        ]


schema = graphene.Schema(query=Query)
