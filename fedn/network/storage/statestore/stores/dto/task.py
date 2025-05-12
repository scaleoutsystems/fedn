from typing import Optional

from google.protobuf.json_format import MessageToDict

import fedn.network.grpc.fedn_pb2 as fedn_proto
from fedn.network.storage.statestore.stores.dto.shared import BaseDTO, Field, PrimaryID


class Task(BaseDTO):
    task_id: str = PrimaryID(None)
    client_id: str = Field(None)
    combiner_id: str = Field(None)

    type: str = Field(None)
    parameters: dict = Field(None)

    model_id: Optional[str] = Field(None)
    round_id: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)


class TaskState(BaseDTO):
    # Not best practive to use the same primary key for two different models
    # but since it is unknown if TaskState will be moved to another store in the future this will do
    task_id: str = PrimaryID(None)

    status: fedn_proto.TaskStatus = Field(None)
    response: dict = Field(None)

    def to_proto(self) -> fedn_proto.TaskState:
        task_state = fedn_proto.TaskState()
        task_state.task_id = self.task_id
        task_state.status = self.status
        task_state.response.update(self.response)
        return task_state

    @classmethod
    def from_proto(cls, proto: fedn_proto.TaskState) -> "TaskState":
        task_state = TaskState()
        task_state.task_id = proto.task_id
        task_state.status = proto.status
        task_state.response = MessageToDict(proto.response, preserving_proto_field_name=True)
        return task_state
