from typing import Optional

from fedn.network.storage.statestore.stores.dto.session import SessionConfigDTO
from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, DictDTO, Field, ListDTO, PrimaryID


class RoundConfigDTO(SessionConfigDTO):
    """RoundConfig data transfer object."""

    session_id: Optional[str] = Field(None)
    task: str = Field(None)
    rounds: int = Field(None)
    round_id: str = Field(None)
    client_settings: Optional[dict] = Field({})


class RoundReduceDTO(DictDTO):
    """RoundReduce data transfer object."""

    time_aggregate_model: Optional[float] = Field(None)
    time_fetch_model: Optional[float] = Field(None)
    time_load_model: Optional[float] = Field(None)


class RoundDataDTO(DictDTO):
    """RoundData data transfer object."""

    time_commit: Optional[float] = Field(None)
    reduce: Optional[RoundReduceDTO] = Field(None)


class AggregationTimeDTO(DictDTO):
    """AggregationTime data transfer object."""

    nr_aggregated_models: Optional[int] = Field(None)
    time_model_aggregation: Optional[float] = Field(None)
    time_model_load: Optional[float] = Field(None)


class RoundCombinerDataDTO(DictDTO):
    """RoundCombinerData data transfer object."""

    aggregation_time: Optional[AggregationTimeDTO] = Field(None)
    nr_expected_updates: Optional[int] = Field(None)
    nr_required_updates: Optional[int] = Field(None)
    time_combination: Optional[float] = Field(None)
    timeout: Optional[float] = Field(None)


class RoundCombinerConfigDTO(RoundConfigDTO):
    """RoundCombinerConfig data transfer object."""

    _job_id: Optional[str] = Field(None)
    round_id: Optional[str] = Field(None)


class RoundCombinerDTO(DictDTO):
    """RoundCombiner data transfer object."""

    round_id: Optional[str] = Field(None)
    model_id: Optional[str] = Field(None)
    name: str = Field(None)
    status: str = Field(None)
    time_exec_training: float = Field(None)
    config: RoundCombinerConfigDTO = Field(None)
    data: RoundCombinerDataDTO = Field(None)


class RoundDTO(BaseDTO):
    """Round data transfer object."""

    round_id: Optional[str] = PrimaryID(None)
    status: Optional[str] = Field(None)
    round_config: Optional[RoundConfigDTO] = Field(None)
    round_data: Optional[RoundDataDTO] = Field(None)
    combiners: ListDTO[RoundCombinerDTO] = Field(ListDTO(RoundCombinerDTO))
