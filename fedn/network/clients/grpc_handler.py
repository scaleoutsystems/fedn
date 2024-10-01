import sys
import time

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger


class GrpcHandler:

    def __init__(self, host: str, port: int, name: str, token: str, combiner_name: str):
        self.metadata = [
            ("client", name),
            ("grpc-server", combiner_name),
            ("authorization", token)
        ]

        def metadata_callback(context, callback):
            callback(self.metadata, None)

        call_credentials = grpc.metadata_call_credentials(metadata_callback)
        channel_credentials = grpc.composite_channel_credentials(
            grpc.ssl_channel_credentials(), call_credentials
        ) if port == 443 else grpc.composite_channel_credentials(
            grpc.local_channel_credentials(), call_credentials
        )

        url = f"{host}:{port}"

        logger.info(f"Connecting (GRPC) to {url}")

        channel = grpc.secure_channel(url, channel_credentials) if port == 443 else grpc.insecure_channel(url)

        self.connectorStub = rpc.ConnectorStub(channel)
        self.combinerStub = rpc.CombinerStub(channel)
        self.modelStub = rpc.ModelServiceStub(channel)

    def send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0):
        logger.info(f"Sending heartbeat to combiner!!! {client_name} {client_id} {update_frequency}")
        heartbeat = fedn.Heartbeat(sender=fedn.Client(name=client_name, role=fedn.WORKER, client_id=client_id))
        try:
            logger.info("Sending heartbeat to combiner")
            self.connectorStub.SendHeartbeat(heartbeat)
            logger.info("Sending heartbeat to combiner 222")
        except grpc.RpcError as e:
            logger.info("Sending heartbeat to combiner 222asdfasdf")
            status_code = e.code()

            if status_code == grpc.StatusCode.UNAVAILABLE:
                logger.error("GRPC hearbeat: combiner unavailable")

            elif status_code == grpc.StatusCode.UNAUTHENTICATED:
                details = e.details()
                if details == "Token expired":
                    logger.error("GRPC hearbeat: Token expired. Disconnecting.")
                    sys.exit("Unauthorized. Token expired. Please obtain a new token.")

        logger.info(f"Sleeping for {update_frequency} seconds")

        time.sleep(update_frequency)

        logger.info(f"Slept for {update_frequency} seconds")

        self.send_heartbeats(client_name=client_name, client_id=client_id, update_frequency=update_frequency)

