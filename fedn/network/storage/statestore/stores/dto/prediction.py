from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, NodeDTO, PrimaryID


class PredictionDTO(BaseDTO):
    """Prediction data transfer object."""

    prediction_id: Optional[str] = PrimaryID(None)
    model_id: str = Field(None)
    data: str = Field(None)
    correlation_id: str = Field(None)
    timestamp: str = Field(None)
    meta: Optional[str] = Field(None)
    sender: NodeDTO = Field(NodeDTO())
    receiver: NodeDTO = Field(NodeDTO())
