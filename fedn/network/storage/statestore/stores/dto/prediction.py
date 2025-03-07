from fedn.network.storage.statestore.stores.dto.shared import AgentDTO, BaseDTO, Field


class PredictionDTO(BaseDTO):
    """Prediction data transfer object."""

    prediction_id: str = Field(None)
    model_id: str = Field(None)
    data: str = Field(None)
    correlation_id: str = Field(None)
    timestamp: str = Field(None)
    meta: str = Field(None)
    sender: AgentDTO = AgentDTO()
    receiver: AgentDTO = AgentDTO()
