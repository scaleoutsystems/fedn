from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field


class AnalyticDTO(BaseDTO):
    id: str = Field(None)
    sender_id: str = Field(None)
    sender_role: str = Field(None)
    memory_utilisation: float = Field(None)
    cpu_utilisation: float = Field(None)
