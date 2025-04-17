from typing import Tuple

from fedn.network.storage.statestore.stores.shared import SortOrder

api_version = "v1"


def is_positive_integer(s):
    return s is not None and s.isdigit() and int(s) > 0


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

    if sort_order is not None:
        sort_order = SortOrder.ASCENDING if sort_order.lower() == "asc" else SortOrder.DESCENDING
    else:
        sort_order = SortOrder.DESCENDING

    return limit, skip, sort_key, sort_order


def get_post_data_to_kwargs(request: object) -> dict:
    try:
        # Try to get data from form
        request_data = request.form.to_dict()
    except Exception:
        request_data = None

    if not request_data:
        try:
            # Try to get data from JSON
            request_data = request.get_json()
        except Exception:
            request_data = {}

    kwargs = {}
    for key, value in request_data.items():
        if isinstance(value, str) and "," in value:
            kwargs[key] = {"$in": value.split(",")}
        else:
            kwargs[key] = value

    return kwargs
