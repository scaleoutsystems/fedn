from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field


class SessionConfigDTO(BaseDTO):
    """SessionConfig data transfer object."""

    id: str = Field(None)
    aggregator: str = Field(None)
    round_timeout: int = Field(None)
    buffer_size: int = Field(None)
    rounds: int = Field(None)
    delete_models_storage: bool = Field(None)
    clients_required: int = Field(None)
    validate: bool = Field(None)
    helper_type: str = Field(None)
    model_id: str = Field(None)
    server_functions: str = Field(None)


class SessionDTO(BaseDTO):
    """Session data transfer object."""

    session_id: str = Field(None)
    name: str = Field(None)
    status: str = Field(None)
    session_config: SessionConfigDTO = Field(None)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "name": self.name,
            "status": self.status,
            "session_config": self.session_config.to_dict(),
        }

    def to_db(self, exclude_unset: bool = False):
        return {
            "session_id": self.session_id,
            "name": self.name,
            "status": self.status,
            "session_config": self.session_config.to_db(exclude_unset=exclude_unset),
        }
