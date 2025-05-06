"""This module provides classes for managing database connections and stores in a federated network environment.

Classes:
    StoreContainer: A container for various store instances.
    DatabaseConnection: A singleton class for managing database connections and stores.
"""

from typing import Type

import pymongo
from pymongo.database import Database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session as SessionClass

from fedn.common.log_config import logger
from fedn.network.storage.statestore.stores.attribute_store import AttributeStore, MongoDBAttributeStore, SQLAttributeStore
from fedn.network.storage.statestore.stores.client_store import ClientStore, MongoDBClientStore, SQLClientStore
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore, MongoDBCombinerStore, SQLCombinerStore
from fedn.network.storage.statestore.stores.metric_store import MetricStore, MongoDBMetricStore, SQLMetricStore
from fedn.network.storage.statestore.stores.model_store import ModelStore, MongoDBModelStore, SQLModelStore
from fedn.network.storage.statestore.stores.package_store import MongoDBPackageStore, PackageStore, SQLPackageStore
from fedn.network.storage.statestore.stores.prediction_store import MongoDBPredictionStore, PredictionStore, SQLPredictionStore
from fedn.network.storage.statestore.stores.round_store import MongoDBRoundStore, RoundStore, SQLRoundStore
from fedn.network.storage.statestore.stores.run_store import MongoDBRunStore, RunStore, SQLRunStore
from fedn.network.storage.statestore.stores.session_store import MongoDBSessionStore, SessionStore, SQLSessionStore
from fedn.network.storage.statestore.stores.sql.shared import MyAbstractBase
from fedn.network.storage.statestore.stores.status_store import MongoDBStatusStore, SQLStatusStore, StatusStore
from fedn.network.storage.statestore.stores.telemetry_store import MongoDBTelemetryStore, SQLTelemetryStore, TelemetryStore
from fedn.network.storage.statestore.stores.validation_store import MongoDBValidationStore, SQLValidationStore, ValidationStore


class DatabaseConnection:
    """Singleton class for managing database connections and stores."""

    _instance = None

    client_store: ClientStore
    validation_store: ValidationStore
    combiner_store: CombinerStore
    status_store: StatusStore
    prediction_store: PredictionStore
    round_store: RoundStore
    package_store: PackageStore
    model_store: ModelStore
    session_store: SessionStore
    metric_store: MetricStore
    attribute_store: AttributeStore
    telemetry_store: TelemetryStore
    run_store: RunStore

    def __init__(self, statestore_config, network_id, connect: bool = True):
        self.type: str = None
        self.mdb: Database = None
        self.Session: sessionmaker = None
        self.type = statestore_config["type"]
        self.statestore_config = statestore_config
        self.network_id = network_id
        self._initialized = False
        if connect:
            self.initialize_connection()

    def initialize_connection(self):
        if self._initialized:
            logger.warning("DatabaseConnection is already initialized.")
            return
        self._initialized = True

        if self.type == "MongoDB":
            logger.info("Connecting to MongoDB")
            mdb: Database = self._setup_mongo(self.statestore_config, self.network_id)

            client_store = MongoDBClientStore(mdb, "network.clients")
            validation_store = MongoDBValidationStore(mdb, "control.validations")
            combiner_store = MongoDBCombinerStore(mdb, "network.combiners")
            status_store = MongoDBStatusStore(mdb, "control.status")
            prediction_store = MongoDBPredictionStore(mdb, "control.predictions")
            round_store = MongoDBRoundStore(mdb, "control.rounds")
            package_store = MongoDBPackageStore(mdb, "control.packages")
            model_store = MongoDBModelStore(mdb, "control.model")
            session_store = MongoDBSessionStore(mdb, "control.sessions")
            metric_store = MongoDBMetricStore(mdb, "control.metrics")
            attribute_store = MongoDBAttributeStore(mdb, "control.attributes")
            telemetry_store = MongoDBTelemetryStore(mdb, "control.telemetry")
            run_store = MongoDBRunStore(mdb, "control.training_runs")

            self.mdb = mdb

        elif self.type in ["SQLite", "PostgreSQL"]:
            logger.info("Connecting to SQL database")
            Session = self._setup_sql(self.statestore_config)  # noqa: N806

            client_store = SQLClientStore(Session)
            validation_store = SQLValidationStore(Session)
            combiner_store = SQLCombinerStore(Session)
            status_store = SQLStatusStore(Session)
            prediction_store = SQLPredictionStore(Session)
            round_store = SQLRoundStore(Session)
            package_store = SQLPackageStore(Session)
            model_store = SQLModelStore(Session)
            session_store = SQLSessionStore(Session)
            metric_store = SQLMetricStore(Session)
            attribute_store = SQLAttributeStore(Session)
            telemetry_store = SQLTelemetryStore(Session)
            run_store = SQLRunStore(Session)
            self.Session = Session

        else:
            raise ValueError("Unknown statestore type")

        self.client_store: ClientStore = client_store
        self.validation_store: ValidationStore = validation_store
        self.combiner_store: CombinerStore = combiner_store
        self.status_store: StatusStore = status_store
        self.prediction_store: PredictionStore = prediction_store
        self.round_store: RoundStore = round_store
        self.package_store: PackageStore = package_store
        self.model_store: ModelStore = model_store
        self.session_store: SessionStore = session_store
        self.metric_store: SQLMetricStore = metric_store
        self.attribute_store: AttributeStore = attribute_store
        self.telemetry_store: TelemetryStore = telemetry_store
        self.run_store: RunStore = run_store

    def _setup_mongo(self, statestore_config: dict, network_id: str) -> Database:
        mc = pymongo.MongoClient(**statestore_config["mongo_config"])
        mc.server_info()
        mdb: Database = mc[network_id]

        return mdb

    def _setup_sql(self, statestore_config: dict) -> Type[SessionClass]:
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
