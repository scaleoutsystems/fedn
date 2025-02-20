"""Shared objects for the network API."""

import os
from typing import Tuple

from werkzeug.security import safe_join

from fedn.common.config import get_modelstorage_config, get_network_config
from fedn.network.controller.control import Control
from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.s3.base import RepositoryBase
from fedn.network.storage.s3.miniorepository import MINIORepository
from fedn.network.storage.s3.repository import Repository
from fedn.network.storage.s3.saasrepository import SAASRepository
from fedn.network.storage.statestore.stores.analytic_store import AnalyticStore
from fedn.network.storage.statestore.stores.client_store import ClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore
from fedn.network.storage.statestore.stores.model_store import ModelStore
from fedn.network.storage.statestore.stores.package_store import PackageStore
from fedn.network.storage.statestore.stores.prediction_store import PredictionStore
from fedn.network.storage.statestore.stores.round_store import RoundStore
from fedn.network.storage.statestore.stores.session_store import SessionStore
from fedn.network.storage.statestore.stores.status_store import StatusStore
from fedn.network.storage.statestore.stores.validation_store import ValidationStore
from fedn.utils.checksum import sha

modelstorage_config = get_modelstorage_config()
network_id = get_network_config()

# TODO: Refactor all access to the stores to use the DatabaseConnection
stores = DatabaseConnection().get_stores()
session_store: SessionStore = stores.session_store
model_store: ModelStore = stores.model_store
round_store: RoundStore = stores.round_store
package_store: PackageStore = stores.package_store
combiner_store: CombinerStore = stores.combiner_store
client_store: ClientStore = stores.client_store
status_store: StatusStore = stores.status_store
validation_store: ValidationStore = stores.validation_store
prediction_store: PredictionStore = stores.prediction_store
analytic_store: AnalyticStore = stores.analytic_store


repository = Repository(modelstorage_config["storage_config"], storage_type=modelstorage_config["storage_type"])

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

storage_type = os.environ.get("FEDN_STORAGE_TYPE", modelstorage_config["storage_type"])
if storage_type == "MINIO":
    minio_repository = MINIORepository(modelstorage_config["storage_config"])
elif storage_type == "SAAS":
    minio_repository = SAASRepository(modelstorage_config["storage_config"])
else:
    minio_repository = MINIORepository(modelstorage_config["storage_config"])


def get_checksum(name: str = None) -> Tuple[bool, str, str]:
    """Generate a checksum for a given file."""
    message = None
    sum = None
    success = False

    if name is None:
        active_package = package_store.get_active()
        if active_package is None:
            message = "No compute package uploaded"
            return success, message, sum
        name = active_package["storage_file_name"]
    file_path = safe_join(os.getcwd(), name)
    try:
        sum = str(sha(file_path))
        success = True
        message = "Checksum created."
    except FileNotFoundError:
        message = "File not found."
    return success, message, sum
