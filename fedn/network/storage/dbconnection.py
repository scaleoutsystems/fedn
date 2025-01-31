import pymongo
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fedn.common.config import get_network_config, get_statestore_config
from fedn.network.storage.statestore.stores.client_store import ClientStore, MongoDBClientStore, SQLClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore, MongoDBCombinerStore, SQLCombinerStore
from fedn.network.storage.statestore.stores.model_store import MongoDBModelStore, SQLModelStore
from fedn.network.storage.statestore.stores.package_store import MongoDBPackageStore, PackageStore, SQLPackageStore
from fedn.network.storage.statestore.stores.prediction_store import MongoDBPredictionStore, PredictionStore, SQLPredictionStore
from fedn.network.storage.statestore.stores.round_store import MongoDBRoundStore, RoundStore, SQLRoundStore
from fedn.network.storage.statestore.stores.session_store import MongoDBSessionStore, SQLSessionStore
from fedn.network.storage.statestore.stores.status_store import MongoDBStatusStore, SQLStatusStore, StatusStore
from fedn.network.storage.statestore.stores.store import MyAbstractBase
from fedn.network.storage.statestore.stores.validation_store import MongoDBValidationStore, SQLValidationStore, ValidationStore


class StoreContainer:
    def __init__(
        self,
        client_store: ClientStore,
        validation_store: ValidationStore,
        combiner_store: CombinerStore,
        status_store: StatusStore,
        prediction_store: PredictionStore,
        round_store: RoundStore,
        package_store: PackageStore,
        model_store: MongoDBModelStore,
        session_store: MongoDBSessionStore,
    ):
        self.client_store = client_store
        self.validation_store = validation_store
        self.combiner_store = combiner_store
        self.status_store = status_store
        self.prediction_store = prediction_store
        self.round_store = round_store
        self.package_store = package_store
        self.model_store = model_store
        self.session_store = session_store


class DatabaseConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._init_connection()
        return cls._instance

    def _init_connection(self):
        statestore_config = get_statestore_config()
        network_id = get_network_config()

        if statestore_config["type"] == "MongoDB":
            mdb: DatabaseConnection = self._setup_mongo(statestore_config, network_id)

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
            Session = self._setup_sql(statestore_config)

            client_store = SQLClientStore(Session)
            validation_store = SQLValidationStore(Session)
            combiner_store = SQLCombinerStore(Session)
            status_store = SQLStatusStore(Session)
            prediction_store = SQLPredictionStore(Session)
            round_store = SQLRoundStore(Session)
            package_store = SQLPackageStore(Session)
            model_store = SQLModelStore(Session)
            session_store = SQLSessionStore(Session)
        else:
            raise ValueError("Unknown statestore type")

        self.sc = StoreContainer(
            client_store, validation_store, combiner_store, status_store, prediction_store, round_store, package_store, model_store, session_store
        )

    def _setup_mongo(self, statestore_config, network_id):
        mc = pymongo.MongoClient(**statestore_config["mongo_config"])
        mc.server_info()
        mdb: DatabaseConnection = mc[network_id]

        return mdb

    def _setup_sql(self, statestore_config):
        if statestore_config["type"] == "SQLite":
            engine = create_engine("sqlite:///my_database.db", echo=True)
        elif statestore_config["type"] == "PostgreSQL":
            postgres_config = statestore_config["postgres_config"]
            username = postgres_config["username"]
            password = postgres_config["password"]
            host = postgres_config["host"]
            port = postgres_config["port"]

            engine = create_engine(f"postgresql://{username}:{password}@{host}:{port}/fedn_db", echo=True)

        Session = sessionmaker(engine)

        MyAbstractBase.metadata.create_all(engine, checkfirst=True)

        return Session

    def get_stores(self):
        return self.sc
