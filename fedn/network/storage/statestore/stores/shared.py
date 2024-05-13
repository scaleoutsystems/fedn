
def from_document(document: dict) -> dict:
    document["id"] = str(document["_id"])
    del document["_id"]
    return document


class EntityNotFound(Exception):
    pass
