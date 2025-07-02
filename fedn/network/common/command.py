from enum import Enum


class CommandType(Enum):
    StandardSession = "Fedn_StandardSession"
    SplitLearningSession = "Fedn_SplitLearningSession"
    PredictionSession = "Fedn_Prediction"


def validate_custom_command(command: str) -> bool:
    """Validate that the command is a valid custom command.

    :param command: The command to validate.
    :return: True if the command is valid, False otherwise.
    """
    if not isinstance(command, str):
        return False
    if not command.startswith("Fedn_"):
        return False
    return True
