from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, DictDTO, Field, PrimaryID, validator
from fedn.network.storage.statestore.stores.shared import ValidationError


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

    @validator
    def validate_config(self):
        if not self.aggregator:
            raise ValidationError("aggregator", "Aggregator is required")

        if not self.round_timeout:
            raise ValidationError("round_timeout", "Round timeout is required")

        if not self.buffer_size:
            raise ValidationError("buffer_size", "Buffer size is required")

        if not self.model_id:
            raise ValidationError("model_id", "Model ID is required")

        if self.delete_models_storage is None:
            raise ValidationError("delete_models_storage", "Delete models storage is required")

        if not self.clients_required:
            raise ValidationError("clients_required", "Clients required is required")

        if self.validate is None:
            raise ValidationError("validate", "Validate is required")

        if not self.helper_type:
            raise ValidationError("helper_type", "Helper type is required")


class SessionDTO(BaseDTO):
    """Session data transfer object."""

    session_id: str = PrimaryID(None)
    name: Optional[str] = Field(None)
    status: Optional[str] = Field("Created")
    session_config: SessionConfigDTO = Field(SessionConfigDTO())
    seed_model_id: Optional[str] = Field(None)
