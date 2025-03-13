from concurrent import futures
from typing import TypedDict

import grpc
from grpc_health.v1 import health, health_pb2_grpc

import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger, set_log_level_from_string, set_log_stream
from fedn.network.combiner.shared import modelservice
from fedn.network.grpc.auth import JWTInterceptor


class ServerConfig(TypedDict):
    port: int
    secure: bool
    key: str
    certificate: str
    logfile: str
    verbosity: str


class Server:
    """Class for configuring and launching the gRPC server."""

    def __init__(self, servicer, config: ServerConfig):
        set_log_level_from_string(config.get("verbosity", "INFO"))
        set_log_stream(config.get("logfile", None))

        # Keepalive settings: these detect if the client is alive
        KEEPALIVE_TIME_MS = 60 * 1000  # send keepalive ping every 60 seconds
        KEEPALIVE_TIMEOUT_MS = 20 * 1000  # wait 20 seconds for keepalive ping ack before considering connection dead
        MAX_CONNECTION_IDLE_MS = 5 * 60 * 1000  # max idle time before server terminates the connection (5 minutes)

        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=350),
            interceptors=[JWTInterceptor()],
            options=[
                ("grpc.keepalive_time_ms", KEEPALIVE_TIME_MS),
                ("grpc.keepalive_timeout_ms", KEEPALIVE_TIMEOUT_MS),
                ("grpc.max_connection_idle_ms", MAX_CONNECTION_IDLE_MS),
            ],
        )
        self.certificate = None
        self.health_servicer = health.HealthServicer()

        if isinstance(servicer, rpc.CombinerServicer):
            rpc.add_CombinerServicer_to_server(servicer, self.server)
        if isinstance(servicer, rpc.ConnectorServicer):
            rpc.add_ConnectorServicer_to_server(servicer, self.server)
        if isinstance(servicer, rpc.ReducerServicer):
            rpc.add_ReducerServicer_to_server(servicer, self.server)
        if isinstance(modelservice, rpc.ModelServiceServicer):
            rpc.add_ModelServiceServicer_to_server(modelservice, self.server)
        if isinstance(servicer, rpc.CombinerServicer):
            rpc.add_ControlServicer_to_server(servicer, self.server)

        health_pb2_grpc.add_HealthServicer_to_server(self.health_servicer, self.server)

        if config["secure"]:
            logger.info("Creating secure gRPCS server using certificate")
            server_credentials = grpc.ssl_server_credentials(
                (
                    (
                        config["key"],
                        config["certificate"],
                    ),
                )
            )
            self.server.add_secure_port("[::]:" + str(config["port"]), server_credentials)
        else:
            logger.info("Creating gRPC server")
            self.server.add_insecure_port("[::]:" + str(config["port"]))

    def start(self):
        """Start the gRPC server."""
        logger.info("gRPC Server started")
        self.server.start()

    def stop(self):
        """Stop the gRPC server."""
        logger.info("gRPC Server stopped")
        self.server.stop(0)
