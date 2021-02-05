
from enum import Enum

class ClientState(Enum):
    idle = 1
    training = 2
    validating = 3


def ClientStateToString(state):
    if state == ClientState.idle:
        return "IDLE"
    if state == ClientState.training:
        return "TRAINING"
    if state == ClientState.validating:
        return "VALIDATING"

    return "UNKNOWN"
