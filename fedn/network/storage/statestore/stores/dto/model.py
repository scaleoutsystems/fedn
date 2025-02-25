from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field


class ModelDTO(BaseDTO):
    """Model data transfer object."""

    model_id: Optional[str] = Field(None)
    name: str = Field(None)
    parent_model: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)

    @property
    def model(self):
        return self.model_id

    @model.setter
    def model(self, value):
        self.model_id = value
