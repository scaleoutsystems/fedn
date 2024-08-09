from enum import Enum


class ReducerState(Enum):
    """Enum for representing the state of a reducer."""

    setup = 1
    idle = 2
    instructing = 3
    monitoring = 4


def ReducerStateToString(state):
    """Convert ReducerState to string.

    :param state: The state.
    :type state: :class:`fedn.network.state.ReducerState`
    :return: The state as string.
    :rtype: str
    """
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
    """Convert string to ReducerState.

    :param state: The state as string.
    :type state: str
    :return: The state.
    :rtype: :class:`fedn.network.state.ReducerState`
    """
    if state == "setup":
        return ReducerState.setup
    if state == "idle":
        return ReducerState.idle
    elif state == "instructing":
        return ReducerState.instructing
    elif state == "monitoring":
        return ReducerState.monitoring
