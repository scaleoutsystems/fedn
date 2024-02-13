import pymongo
from pymongo.database import Database

from fedn.common.config import get_network_config, get_statestore_config

api_version = "v1"

statestore_config = get_statestore_config()
network_id = get_network_config()

mc = pymongo.MongoClient(**statestore_config["mongo_config"])
mc.server_info()
mdb: Database = mc[network_id]
