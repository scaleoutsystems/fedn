from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, PrimaryID


class AnalyticDTO(BaseDTO):
    id: str = PrimaryID(None)
    sender_id: str = Field(None)
    sender_role: str = Field(None)
    memory_utilisation: float = Field(None)
    cpu_utilisation: float = Field(None)


# valid, msg = _validate_analytic(item_dict)

# def _validate_analytic(analytic: dict) -> Tuple[bool, str]:
#     if "sender_id" not in analytic:
#         return False, "sender_id is required"
#     if "sender_role" not in analytic or analytic["sender_role"] not in ["combiner", "client"]:
#         return False, "sender_role must be either 'combiner' or 'client'"
#     return analytic, ""
