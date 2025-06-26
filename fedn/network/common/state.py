from enum import Enum


class ControllerState(Enum):
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
    if state == ControllerState.setup:
        return "setup"
    if state == ControllerState.idle:
        return "idle"
    if state == ControllerState.instructing:
        return "instructing"
    if state == ControllerState.monitoring:
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
        return ControllerState.setup
    if state == "idle":
        return ControllerState.idle
    elif state == "instructing":
        return ControllerState.instructing
    elif state == "monitoring":
        return ControllerState.monitoring
