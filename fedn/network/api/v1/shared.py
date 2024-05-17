from typing import Tuple

import pymongo
from pymongo.database import Database

from fedn.network.api.shared import statestore_config, network_id

api_version = "v1"

mc = pymongo.MongoClient(**statestore_config["mongo_config"])
mc.server_info()
mdb: Database = mc[network_id]


def is_positive_integer(s):
    return s is not None and s.isdigit() and int(s) > 0


def get_use_typing(headers: object) -> bool:
    skip_typing: str = headers.get("X-Skip-Typing", "false")
    return False if skip_typing.lower() == "true" else True


def get_limit(headers: object) -> int:
    limit: str = headers.get("X-Limit")
    if is_positive_integer(limit):
        return int(limit)
    return 0


def get_reverse(headers: object) -> bool:
    reverse: str = headers.get("X-Reverse")
    if reverse and reverse.lower() == "true":
        return True
    return False


def get_skip(headers: object) -> int:
    skip: str | None = headers.get("X-Skip")
    if is_positive_integer(skip):
        return int(skip)
    return 0


def get_typed_list_headers(
    headers: object,
) -> Tuple[int, int, str, int, bool]:
    sort_key: str = headers.get("X-Sort-Key")
    sort_order: str = headers.get("X-Sort-Order")

    limit: int = get_limit(headers)
    skip: int = get_skip(headers)
    use_typing: bool = get_use_typing(headers)

    if sort_order is not None:
        sort_order = pymongo.ASCENDING if sort_order.lower() == "asc" else pymongo.DESCENDING
    else:
        sort_order = pymongo.DESCENDING

    return limit, skip, sort_key, sort_order, use_typing


def get_post_data_to_kwargs(request: object) -> dict:
    request_data = request.form.to_dict()

    if not request_data:
        request_data = request.json

    kwargs = {}
    for key, value in request_data.items():
        if "," in value:
            kwargs[key] = {"$in": value.split(",")}
        else:
            kwargs[key] = value

    return kwargs
