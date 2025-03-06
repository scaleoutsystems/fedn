from typing import Optional

from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field


class IdentityDTO(BaseDTO):
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
    sender: IdentityDTO = Field(IdentityDTO())
    receiver: IdentityDTO = Field(IdentityDTO())

    def to_dict(self):
        output = super().to_dict()
        if self.sender:
            output["sender"] = self.sender.to_dict()
        if self.receiver:
            output["receiver"] = self.receiver.to_dict()
        return output

    def to_db(self, exclude_unset=False):
        obj = super().to_db(exclude_unset=exclude_unset)

        if self.sender:
            obj["sender"] = self.sender.to_db(exclude_unset=exclude_unset)
        if self.receiver:
            obj["receiver"] = self.receiver.to_db(exclude_unset=exclude_unset)

        return obj

    def populate_with(self, data):
        if "sender" in data:
            sender = data.pop("sender")
            if sender is not None:
                self.sender = IdentityDTO(**sender)
            else:
                self.sender = IdentityDTO()
        else:
            raise ValueError("Missing key: sender")
        if "receiver" in data:
            receiver = data.pop("receiver")
            if receiver is not None:
                self.receiver = IdentityDTO(**receiver)
            else:
                self.receiver = IdentityDTO()
        else:
            raise ValueError("Missing key: reciever")

        super().populate_with(data)
