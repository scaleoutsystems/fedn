import copy
import uuid
from datetime import datetime

import pymongo
from google.protobuf.json_format import MessageToDict

from fedn.common.log_config import logger
from fedn.network.state import ReducerStateToString, StringToReducerState


class MongoStateStore:
    """Statestore implementation using MongoDB.

    :param network_id: The network id.
    :type network_id: str
    :param config: The statestore configuration.
    :type config: dict
    :param defaults: The default configuration. Given by config/settings-reducer.yaml.template
    :type defaults: dict
    """

    def __init__(self, network_id, config):
        """Constructor."""
        self.__inited = False
