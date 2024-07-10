from fedn.common.config import get_modelstorage_config, get_network_config, get_statestore_config
from fedn.network.controller.control import Control
from fedn.network.storage.statestore.mongostatestore import MongoStateStore
statestore=None
statestore_config=None
modelstorage_config=None
network_id=None
control=None
def set_statestore_config():
    global statestore
    global modelstorage_config
    global network_id
    global control
    global statestore_config
    statestore_config = get_statestore_config()
    modelstorage_config = get_modelstorage_config()
    network_id = get_network_config()
    statestore = MongoStateStore(network_id, statestore_config["mongo_config"])
    statestore.set_storage_backend(modelstorage_config)
    control = Control(statestore=statestore)

set_statestore_config()
