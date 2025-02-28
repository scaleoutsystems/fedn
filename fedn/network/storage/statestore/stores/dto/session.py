from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, OptionalField


class SessionConfigDTO(BaseDTO):
    """SessionConfig data transfer object."""

    aggregator: str = Field(None)
    round_timeout: int = Field(None)
    buffer_size: int = Field(None)
    rounds: Optional[int] = OptionalField(None)
    delete_models_storage: bool = Field(None)
    clients_required: int = Field(None)
    validate: bool = Field(None)
    helper_type: str = Field(None)
    model_id: str = Field(None)
    server_functions: Optional[str] = OptionalField(None)


class SessionDTO(BaseDTO):
    """Session data transfer object."""

    session_id: str = Field(None)
    name: str = Field(None)
    status: Optional[str] = OptionalField("Created")
    session_config: SessionConfigDTO = Field(None)

    def to_dict(self):
        obj = super().to_dict()

        if self.session_config:
            obj["session_config"] = self.session_config.to_dict()

        return obj

    def to_db(self, exclude_unset: bool = False):
        obj = super().to_db(exclude_unset=exclude_unset)

        if self.session_config:
            obj["session_config"] = self.session_config.to_db(exclude_unset=exclude_unset)

        return obj
