import json
from typing import Optional

import fedn.network.grpc.fedn_pb2 as fedn


class LoggingContext:
    """Context for keeping track of the session, model and round IDs during a dispatched call from a request."""

    def __init__(
        self, *, step: int = 0, model_id: str = None, round_id: str = None, session_id: str = None, request: Optional[fedn.TaskRequest] = None
    ) -> None:
        if request is not None:
            if model_id is None:
                model_id = request.model_id
            if round_id is None:
                if request.type == fedn.StatusType.MODEL_UPDATE:
                    config = json.loads(request.data)
                    round_id = config["round_id"]
            if session_id is None:
                session_id = request.session_id

        self.model_id = model_id
        self.round_id = round_id
        self.session_id = session_id
        self.request = request
        self.step = step
