from enum import Enum


class ClientRequestType(Enum):
    Connect = "Connect"
    Disconnect = "Disconnect"
    VersionCheck = "VersionCheck"
