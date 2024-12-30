import pymongo
from pymongo.database import Database

from fedn.common.config import get_modelstorage_config, get_network_config, get_statestore_config
from fedn.network.controller.control import Control
from fedn.network.storage.s3.base import RepositoryBase
from fedn.network.storage.s3.miniorepository import MINIORepository
from fedn.network.storage.s3.repository import Repository
from fedn.network.storage.statestore.mongostatestore import MongoStateStore
from fedn.network.storage.statestore.stores.client_store import ClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore
from fedn.network.storage.statestore.stores.model_store import MongoDBModelStore
from fedn.network.storage.statestore.stores.package_store import MongoDBPackageStore
from fedn.network.storage.statestore.stores.round_store import RoundStore
from fedn.network.storage.statestore.stores.session_store import SessionStore
from fedn.network.storage.statestore.stores.status_store import StatusStore
from fedn.network.storage.statestore.stores.validation_store import ValidationStore

statestore_config = get_statestore_config()
modelstorage_config = get_modelstorage_config()
network_id = get_network_config()
statestore = MongoStateStore(network_id, statestore_config["mongo_config"])
# statestore.set_storage_backend(modelstorage_config)

mc = pymongo.MongoClient(**statestore_config["mongo_config"])
mc.server_info()
mdb: Database = mc[network_id]

client_store = ClientStore(mdb, "network.clients")
package_store = MongoDBPackageStore(mdb, "control.package")
session_store = SessionStore(mdb, "control.sessions")
model_store = MongoDBModelStore(mdb, "control.model")
combiner_store = CombinerStore(mdb, "network.combiners")
round_store = RoundStore(mdb, "control.rounds")
status_store = StatusStore(mdb, "control.status")
validation_store = ValidationStore(mdb, "control.validations")

repository = Repository(modelstorage_config["storage_config"])

control = Control(
    network_id=network_id,
    session_store=session_store,
    model_store=model_store,
    round_store=round_store,
    package_store=package_store,
    combiner_store=combiner_store,
    client_store=client_store,
    model_repository=repository,
)

# TODO: use Repository
minio_repository: RepositoryBase = None

if modelstorage_config["storage_type"] == "S3":
    minio_repository = MINIORepository(modelstorage_config["storage_config"])


storage_collection = mdb["network.storage"]

# storage_config = storage_collection.find_one({"status": "enabled"}, projection={"_id": False})

# repository: RepositoryBase = None

# if storage_config["storage_type"] == "S3":
#     repository = Repository(storage_config["storage_config"])
