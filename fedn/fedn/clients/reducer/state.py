from enum import Enum


class ReducerState(Enum):
    idle = 1
    instructing = 2
    monitoring = 3


def ReducerStateToString(state):
    if state == ReducerState.idle:
        return "idle"
    if state == ReducerState.instructing:
        return "instructing"
    if state == ReducerState.monitoring:
        return "monitoring"

    return "UNKNOWN"


def StringToReducerState(state):
    if state == "idle":
        return ReducerState.idle
    elif state == "instructing":
        return ReducerState.instructing
    elif state == "monitoring":
        return ReducerState.monitoring
