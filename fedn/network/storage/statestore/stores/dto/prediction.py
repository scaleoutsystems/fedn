from fedn.network.storage.statestore.stores.dto.shared import DTO, BaseDTO, Field


class IdentityDTO(DTO):
    """Identity data transfer object."""

    name: str = Field(None)
    role: str = Field(None)


class PredictionDTO(BaseDTO):
    """Prediction data transfer object."""

    prediction_id: str = Field(None)
    model_id: str = Field(None)
    data: str = Field(None)
    correlation_id: str = Field(None)
    timestamp: str = Field(None)
    meta: str = Field(None)
    sender: IdentityDTO = IdentityDTO()
    receiver: IdentityDTO = IdentityDTO()
