import os

import pymongo
from pymongo.database import Database
from werkzeug.security import safe_join

from fedn.common.config import get_modelstorage_config, get_network_config, get_statestore_config
from fedn.network.controller.control import Control
from fedn.network.storage.s3.base import RepositoryBase
from fedn.network.storage.s3.miniorepository import MINIORepository
from fedn.network.storage.s3.repository import Repository
from fedn.network.storage.statestore.stores.client_store import ClientStore, MongoDBClientStore, SQLClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore, MongoDBCombinerStore, SQLCombinerStore
from fedn.network.storage.statestore.stores.model_store import MongoDBModelStore, SQLModelStore
from fedn.network.storage.statestore.stores.package_store import MongoDBPackageStore, PackageStore, SQLPackageStore
from fedn.network.storage.statestore.stores.prediction_store import MongoDBPredictionStore, PredictionStore, SQLPredictionStore
from fedn.network.storage.statestore.stores.round_store import MongoDBRoundStore, RoundStore, SQLRoundStore
from fedn.network.storage.statestore.stores.session_store import MongoDBSessionStore, SQLSessionStore
from fedn.network.storage.statestore.stores.shared import EntityNotFound
from fedn.network.storage.statestore.stores.status_store import MongoDBStatusStore, SQLStatusStore, StatusStore
from fedn.network.storage.statestore.stores.store import MyAbstractBase, engine
from fedn.network.storage.statestore.stores.validation_store import MongoDBValidationStore, SQLValidationStore, ValidationStore
from fedn.utils.checksum import sha

statestore_config = get_statestore_config()
modelstorage_config = get_modelstorage_config()
network_id = get_network_config()

client_store: ClientStore = None
validation_store: ValidationStore = None
combiner_store: CombinerStore = None
status_store: StatusStore = None
prediction_store: PredictionStore = None
round_store: RoundStore = None
package_store: PackageStore = None
model_store: SQLModelStore = None
session_store: SQLSessionStore = None

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
    package_store = MongoDBPackageStore(mdb, "control.packages")
    model_store = MongoDBModelStore(mdb, "control.models")
    session_store = MongoDBSessionStore(mdb, "control.sessions")

elif statestore_config["type"] in ["SQLite", "PostgreSQL"]:
    MyAbstractBase.metadata.create_all(engine, checkfirst=True)

    client_store = SQLClientStore()
    validation_store = SQLValidationStore()
    combiner_store = SQLCombinerStore()
    status_store = SQLStatusStore()
    prediction_store = SQLPredictionStore()
    round_store = SQLRoundStore()
    package_store = SQLPackageStore()
    model_store = SQLModelStore()
    session_store = SQLSessionStore()
else:
    raise ValueError("Unknown statestore type")


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


def get_checksum(name: str = None):
    message = None
    sum = None
    success = False

    if name is None:
        try:
            active_package = package_store.get_active()
            name = active_package["storage_file_name"]
        except EntityNotFound:
            message = "No compute package uploaded"
            return success, message, sum
    file_path = safe_join(os.getcwd(), name)
    try:
        sum = str(sha(file_path))
        success = True
        message = "Checksum created."
    except FileNotFoundError:
        message = "File not found."
    return success, message, sum
