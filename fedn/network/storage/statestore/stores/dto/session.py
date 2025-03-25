from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, DictDTO, Field


class SessionConfigDTO(DictDTO):
    """SessionConfig data transfer object."""

    aggregator: str = Field(None)
    aggregator_kwargs: Optional[str] = Field(None)
    round_timeout: int = Field(None)
    buffer_size: int = Field(None)
    rounds: Optional[int] = Field(None)
    delete_models_storage: bool = Field(None)
    clients_required: int = Field(None)
    requested_clients: Optional[int] = Field(None)
    validate: bool = Field(None)
    helper_type: str = Field(None)
    model_id: str = Field(None)
    server_functions: Optional[str] = Field(None)


class SessionDTO(BaseDTO):
    """Session data transfer object."""

    session_id: str = Field(None)
    name: str = Field(None)
    status: Optional[str] = Field("Created")
    session_config: SessionConfigDTO = Field(SessionConfigDTO())


# def validate_session_config(session_config: SessionConfigDTO) -> Tuple[bool, str]:
#     if "aggregator" not in session_config:
#         return False, "session_config.aggregator is required"

#     if "round_timeout" not in session_config:
#         return False, "session_config.round_timeout is required"

#     if not isinstance(session_config["round_timeout"], (int, float)):
#         return False, "session_config.round_timeout must be an integer"

#     if "buffer_size" not in session_config:
#         return False, "session_config.buffer_size is required"

#     if not isinstance(session_config["buffer_size"], int):
#         return False, "session_config.buffer_size must be an integer"

#     if "model_id" not in session_config or session_config["model_id"] == "":
#         return False, "session_config.model_id is required"

#     if not isinstance(session_config["model_id"], str):
#         return False, "session_config.model_id must be a string"

#     if "delete_models_storage" not in session_config:
#         return False, "session_config.delete_models_storage is required"

#     if not isinstance(session_config["delete_models_storage"], bool):
#         return False, "session_config.delete_models_storage must be a boolean"

#     if "clients_required" not in session_config:
#         return False, "session_config.clients_required is required"

#     if not isinstance(session_config["clients_required"], int):
#         return False, "session_config.clients_required must be an integer"

#     if "validate" not in session_config:
#         return False, "session_config.validate is required"

#     if not isinstance(session_config["validate"], bool):
#         return False, "session_config.validate must be a boolean"

#     if "helper_type" not in session_config or session_config["helper_type"] == "":
#         return False, "session_config.helper_type is required"

#     if not isinstance(session_config["helper_type"], str):
#         return False, "session_config.helper_type must be a string"

#     return True, ""


# def validate(item: dict) -> Tuple[bool, str]:
#     if "session_config" not in item or item["session_config"] is None:
#         return False, "session_config is required"

#     session_config = None

#     if isinstance(item["session_config"], dict):
#         session_config = item["session_config"]
#     elif isinstance(item["session_config"], list):
#         session_config = item["session_config"][0]
#     else:
#         return False, "session_config must be a dict"

#     return validate_session_config(session_config)
