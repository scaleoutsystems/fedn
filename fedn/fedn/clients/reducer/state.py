from enum import Enum


class ReducerState(Enum):
    setup = 1
    idle = 2
    instructing = 3
    monitoring = 4


def ReducerStateToString(state):
    if state == ReducerState.setup:
        return "setup"
    if state == ReducerState.idle:
        return "idle"
    if state == ReducerState.instructing:
        return "instructing"
    if state == ReducerState.monitoring:
        return "monitoring"

    return "UNKNOWN"


def StringToReducerState(state):
    if state == "setup":
        return ReducerState.setup
    if state == "idle":
        return ReducerState.idle
    elif state == "instructing":
        return ReducerState.instructing
    elif state == "monitoring":
        return ReducerState.monitoring
