from fedn.common.config import get_modelstorage_config, get_network_config, get_statestore_config
from fedn.network.combiner.modelservice import ModelService
from fedn.network.storage.s3.repository import Repository
from fedn.network.storage.statestore.mongostatestore import MongoStateStore

statestore_config = get_statestore_config()
modelstorage_config = get_modelstorage_config()
network_id = get_network_config()

statestore = MongoStateStore(network_id, statestore_config["mongo_config"])
repository = Repository(modelstorage_config["storage_config"], init_buckets=False)

modelservice = ModelService()
