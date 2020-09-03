from enum import Enum

class ReducerState(Enum):
    idle = 1
    instructing = 2
    monitoring = 3


def ReducerStateToString(state):
    if state == ReducerState.idle:
        return "IDLE"
    if state == ReducerState.instructing:
        return "instructing"
    if state == ReducerState.monitoring:
        return "monitoring"

    return "UNKNOWN"