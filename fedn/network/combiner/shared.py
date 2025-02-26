from fedn.common.config import get_modelstorage_config
from fedn.network.combiner.modelservice import ModelService
from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.s3.repository import Repository
from fedn.network.storage.statestore.stores.client_store import ClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore
from fedn.network.storage.statestore.stores.model_store import ModelStore
from fedn.network.storage.statestore.stores.package_store import PackageStore
from fedn.network.storage.statestore.stores.prediction_store import PredictionStore
from fedn.network.storage.statestore.stores.round_store import RoundStore
from fedn.network.storage.statestore.stores.session_store import SessionStore
from fedn.network.storage.statestore.stores.status_store import StatusStore
from fedn.network.storage.statestore.stores.validation_store import ValidationStore

modelstorage_config = get_modelstorage_config()

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


repository = Repository(modelstorage_config["storage_config"], init_buckets=False)

modelservice = ModelService()
