import pymongo
from pymongo.database import Database

from fedn.common.config import get_modelstorage_config, get_network_config, get_statestore_config
from fedn.network.combiner.modelservice import ModelService
from fedn.network.storage.s3.repository import Repository
from fedn.network.storage.statestore.stores.client_store import ClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore, MongoDBCombinerStore, SQLCombinerStore
from fedn.network.storage.statestore.stores.prediction_store import PredictionStore
from fedn.network.storage.statestore.stores.round_store import MongoDBRoundStore, RoundStore, SQLRoundStore
from fedn.network.storage.statestore.stores.status_store import MongoDBStatusStore, SQLStatusStore, StatusStore
from fedn.network.storage.statestore.stores.store import Base, MyAbstractBase, engine
from fedn.network.storage.statestore.stores.validation_store import ValidationStore

statestore_config = get_statestore_config()
modelstorage_config = get_modelstorage_config()
network_id = get_network_config()

if statestore_config["type"] == "MongoDB":
    mc = pymongo.MongoClient(**statestore_config["mongo_config"])
    mc.server_info()
    mdb: Database = mc[network_id]

MyAbstractBase.metadata.create_all(engine)

client_store = ClientStore(mdb, "network.clients")
validation_store = ValidationStore(mdb, "control.validations")
# combiner_store: CombinerStore = MongoDBCombinerStore(mdb, "network.combiners")
combiner_store: CombinerStore = SQLCombinerStore()
# status_store: StatusStore = MongoDBStatusStore(mdb, "control.status")
status_store: StatusStore = SQLStatusStore()
prediction_store = PredictionStore(mdb, "control.predictions")
# round_store: RoundStore = MongoDBRoundStore(mdb, "control.rounds")
round_store: RoundStore = SQLRoundStore()

repository = Repository(modelstorage_config["storage_config"], init_buckets=False)

modelservice = ModelService()
