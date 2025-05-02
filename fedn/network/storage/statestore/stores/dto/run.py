from datetime import datetime
from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, PrimaryID


class RunDTO(BaseDTO):
    """Training run data transfer object."""

    run_id: Optional[str] = PrimaryID(None)
    session_id: str = Field(None)
    model_id: str = Field(None)
    completed_at_model_id: Optional[str] = Field(None)  # active model id when training run was completed
    round_timeout: int = Field(None)
    rounds: Optional[int] = Field(None)
    completed_at: Optional[datetime] = Field(None)
