from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field


class PredictionDTO(BaseDTO):
    """Prediction data transfer object."""

    model_id: str = Field(None)
    data: str = Field(None)
    correlation_id: str = Field(None)
    timestamp: str = Field(None)
    prediction_id: str = Field(None)
    meta: str = Field(None)
    sender_name: str = Field(None)
    sender_role: str = Field(None)
    receiver_name: str = Field(None)
    receiver_role: str = Field(None)

    @property
    def sender(self):
        return {"name": self.sender_name, "role": self.sender_role}

    @sender.setter
    def sender(self, value: dict):
        if value:
            self.sender_name = value.get("name")
            self.sender_role = value.get("role")
        else:
            self.sender_name = ""
            self.sender_role = ""

    @property
    def receiver(self):
        return {"name": self.receiver_name, "role": self.receiver_role}

    @receiver.setter
    def receiver(self, value: dict):
        if value:
            self.receiver_name = value.get("name")
            self.receiver_role = value.get("role")
        else:
            self.receiver_name = ""
            self.receiver_role = ""
