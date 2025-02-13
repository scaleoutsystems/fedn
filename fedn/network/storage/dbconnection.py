"""This module provides classes for managing database connections and stores in a federated network environment.

Classes:
    StoreContainer: A container for various store instances.
    DatabaseConnection: A singleton class for managing database connections and stores.
"""

import pymongo
from pymongo.database import Database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fedn.common.config import get_network_config, get_statestore_config
from fedn.network.storage.statestore.stores.analytic_store import AnalyticStore, MongoDBAnalyticStore
from fedn.network.storage.statestore.stores.client_store import ClientStore, MongoDBClientStore, SQLClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore, MongoDBCombinerStore, SQLCombinerStore
from fedn.network.storage.statestore.stores.model_store import ModelStore, MongoDBModelStore, SQLModelStore
from fedn.network.storage.statestore.stores.package_store import MongoDBPackageStore, PackageStore, SQLPackageStore
from fedn.network.storage.statestore.stores.prediction_store import MongoDBPredictionStore, PredictionStore, SQLPredictionStore
from fedn.network.storage.statestore.stores.round_store import MongoDBRoundStore, RoundStore, SQLRoundStore
from fedn.network.storage.statestore.stores.session_store import MongoDBSessionStore, SessionStore, SQLSessionStore
from fedn.network.storage.statestore.stores.status_store import MongoDBStatusStore, SQLStatusStore, StatusStore
from fedn.network.storage.statestore.stores.store import MyAbstractBase
from fedn.network.storage.statestore.stores.validation_store import MongoDBValidationStore, SQLValidationStore, ValidationStore


class StoreContainer:
    """A container for various store instances."""

    def __init__(  # noqa: PLR0913
        self,
        client_store: ClientStore,
        validation_store: ValidationStore,
        combiner_store: CombinerStore,
        status_store: StatusStore,
        prediction_store: PredictionStore,
        round_store: RoundStore,
        package_store: PackageStore,
        model_store: ModelStore,
        session_store: SessionStore,
        analytic_store: AnalyticStore,
    ) -> None:
        """Initialize the StoreContainer with various store instances."""
        self.client_store = client_store
        self.validation_store = validation_store
        self.combiner_store = combiner_store
        self.status_store = status_store
        self.prediction_store = prediction_store
        self.round_store = round_store
        self.package_store = package_store
        self.model_store = model_store
        self.session_store = session_store
        self.analytic_store = analytic_store


class DatabaseConnection:
    """Singleton class for managing database connections and stores."""

    _instance = None

    def __new__(cls, *, force_create_new: bool = False) -> "DatabaseConnection":
        """Create a new instance of DatabaseConnection or return the existing singleton instance.

        Args:
            force_create_new (bool): If True, a new instance will be created regardless of the singleton pattern. Only used for testing purpose.

        Returns:
            DatabaseConnection: A new instance if force_create_new is True, otherwise the existing singleton instance.

        """
        if cls._instance is None or force_create_new:
            obj = super(DatabaseConnection, cls).__new__(cls)
            obj._init_connection()
            cls._instance = obj

        return cls._instance

    def _init_connection(self) -> None:
        statestore_config = get_statestore_config()
        network_id = get_network_config()

        if statestore_config["type"] == "MongoDB":
            mdb: Database = self._setup_mongo(statestore_config, network_id)

            client_store = MongoDBClientStore(mdb, "network.clients")
            validation_store = MongoDBValidationStore(mdb, "control.validations")
            combiner_store = MongoDBCombinerStore(mdb, "network.combiners")
            status_store = MongoDBStatusStore(mdb, "control.status")
            prediction_store = MongoDBPredictionStore(mdb, "control.predictions")
            round_store = MongoDBRoundStore(mdb, "control.rounds")
            package_store = MongoDBPackageStore(mdb, "control.packages")
            model_store = MongoDBModelStore(mdb, "control.models")
            session_store = MongoDBSessionStore(mdb, "control.sessions")
            analytic_store = MongoDBAnalyticStore(mdb, "control.analytics")

        elif statestore_config["type"] in ["SQLite", "PostgreSQL"]:
            Session = self._setup_sql(statestore_config)  # noqa: N806

            client_store = SQLClientStore(Session)
            validation_store = SQLValidationStore(Session)
            combiner_store = SQLCombinerStore(Session)
            status_store = SQLStatusStore(Session)
            prediction_store = SQLPredictionStore(Session)
            round_store = SQLRoundStore(Session)
            package_store = SQLPackageStore(Session)
            model_store = SQLModelStore(Session)
            session_store = SQLSessionStore(Session)
            analytic_store = None
        else:
            raise ValueError("Unknown statestore type")

        self.sc = StoreContainer(
            client_store,
            validation_store,
            combiner_store,
            status_store,
            prediction_store,
            round_store,
            package_store,
            model_store,
            session_store,
            analytic_store,
        )

    def close(self) -> None:
        """Close the database connection."""
        pass

    def _setup_mongo(self, statestore_config: dict, network_id: str) -> "DatabaseConnection":
        mc = pymongo.MongoClient(**statestore_config["mongo_config"])
        mc.server_info()
        mdb: Database = mc[network_id]

        return mdb

    def _setup_sql(self, statestore_config: dict) -> "DatabaseConnection":
        if statestore_config["type"] == "SQLite":
            sqlite_config = statestore_config["sqlite_config"]
            dbname = sqlite_config["dbname"]
            engine = create_engine(f"sqlite:///{dbname}", echo=False)
        elif statestore_config["type"] == "PostgreSQL":
            postgres_config = statestore_config["postgres_config"]
            username = postgres_config["username"]
            password = postgres_config["password"]
            host = postgres_config["host"]
            port = postgres_config["port"]

            engine = create_engine(f"postgresql://{username}:{password}@{host}:{port}/fedn_db", echo=False)

        Session = sessionmaker(engine)  # noqa: N806

        MyAbstractBase.metadata.create_all(engine, checkfirst=True)

        return Session

    def get_stores(self) -> StoreContainer:
        """Get the StoreContainer instance."""
        return self.sc

    @property
    def client_store(self) -> ClientStore:
        return self.sc.client_store

    @property
    def validation_store(self) -> ValidationStore:
        return self.sc.validation_store

    @property
    def combiner_store(self) -> CombinerStore:
        return self.sc.combiner_store

    @property
    def status_store(self) -> StatusStore:
        return self.sc.status_store

    @property
    def prediction_store(self) -> PredictionStore:
        return self.sc.prediction_store

    @property
    def round_store(self) -> RoundStore:
        return self.sc.round_store

    @property
    def package_store(self) -> PackageStore:
        return self.sc.package_store

    @property
    def model_store(self) -> ModelStore:
        return self.sc.model_store

    @property
    def session_store(self) -> SessionStore:
        return self.sc.session_store

    @property
    def analytic_store(self) -> AnalyticStore:
        return self.sc.analytic_store
