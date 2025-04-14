from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, PrimaryID


class TrainingRunDTO(BaseDTO):
    """Training run data transfer object."""

    training_run_id: Optional[str] = PrimaryID(None)
    session_id: str = Field(None)
    model_id: str = Field(None)
    round_timeout: int = Field(None)
    rounds: Optional[int] = Field(None)
    completed_at: Optional[datetime] = Field(None)
