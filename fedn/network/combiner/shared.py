import pymongo
from pymongo.database import Database

from fedn.common.config import get_modelstorage_config, get_network_config, get_statestore_config
from fedn.network.combiner.modelservice import ModelService
from fedn.network.storage.s3.repository import Repository
from fedn.network.storage.statestore.stores.client_store import ClientStore, MongoDBClientStore, SQLClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore, MongoDBCombinerStore, SQLCombinerStore
from fedn.network.storage.statestore.stores.prediction_store import MongoDBPredictionStore, PredictionStore, SQLPredictionStore
from fedn.network.storage.statestore.stores.round_store import MongoDBRoundStore, RoundStore, SQLRoundStore
from fedn.network.storage.statestore.stores.status_store import MongoDBStatusStore, SQLStatusStore, StatusStore
from fedn.network.storage.statestore.stores.validation_store import MongoDBValidationStore, SQLValidationStore, ValidationStore

statestore_config = get_statestore_config()
modelstorage_config = get_modelstorage_config()
network_id = get_network_config()

client_store: ClientStore = None
validation_store: ValidationStore = None
combiner_store: CombinerStore = None
status_store: StatusStore = None
prediction_store: PredictionStore = None
round_store: RoundStore = None

if statestore_config["type"] == "MongoDB":
    mc = pymongo.MongoClient(**statestore_config["mongo_config"])
    mc.server_info()
    mdb: Database = mc[network_id]

    client_store = MongoDBClientStore(mdb, "network.clients")
    validation_store = MongoDBValidationStore(mdb, "control.validations")
    combiner_store = MongoDBCombinerStore(mdb, "network.combiners")
    status_store = MongoDBStatusStore(mdb, "control.status")
    prediction_store = MongoDBPredictionStore(mdb, "control.predictions")
    round_store = MongoDBRoundStore(mdb, "control.rounds")
elif statestore_config["type"] in ["SQLite", "PostgreSQL"]:
    client_store = SQLClientStore()
    validation_store = SQLValidationStore()
    combiner_store = SQLCombinerStore()
    status_store = SQLStatusStore()
    prediction_store = SQLPredictionStore()
    round_store = SQLRoundStore()
else:
    raise ValueError("Unknown statestore type")

repository = Repository(modelstorage_config["storage_config"], init_buckets=False)

modelservice = ModelService()
