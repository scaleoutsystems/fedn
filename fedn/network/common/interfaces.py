import base64
import copy
import json
from typing import Dict
import time 

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger
from fedn.network.common.state import ControllerState


class CombinerUnavailableError(Exception):
    pass


class Channel:
    """Wrapper for a gRPC channel.

    :param address: The address for the gRPC server.
    :type address: str
    :param port: The port for connecting to the gRPC server.
    :type port: int
    :param certificate: The certificate for connecting to the gRPC server (optional)
    :type certificate: str
    """

    def __init__(self, address, port, certificate=None):
        """Create a channel.

        If a valid certificate is given, a secure channel is created, else insecure.

        :parameter address: The address for the gRPC server.
        :type address: str
        :parameter port: The port for connecting to the gRPC server.
        :type port: int
        :parameter certificate: The certificate for connecting to the gRPC server (optional)
        :type certificate: str
        """
        self.address = address
        self.port = port
        self.certificate = certificate

        if self.certificate:
            credentials = grpc.ssl_channel_credentials(root_certificates=copy.deepcopy(certificate))
            self.channel = grpc.secure_channel("{}:{}".format(self.address, str(self.port)), credentials)
        else:
            self.channel = grpc.insecure_channel("{}:{}".format(self.address, str(self.port)))

    def get_channel(self):
        """Get a channel.

        :return: An instance of a gRPC channel
        :rtype: :class:`grpc.Channel`
        """
        return copy.copy(self.channel)


class CombinerInterface:
    """Interface for the Combiner (aggregation server).
        Abstraction on top of the gRPC server servicer.

    :param parent: The parent combiner (controller)
    :type parent: :class:`fedn.network.api.interfaces.API`
    :param name: The name of the combiner.
    :type name: str
    :param address: The address of the combiner.
    :type address: str
    :param fqdn: The fully qualified domain name of the combiner.
    :type fqdn: str
    :param port: The port of the combiner.
    :type port: int
    :param certificate: The certificate of the combiner (optional).
    :type certificate: str
    :param key: The key of the combiner (optional).
    :type key: str
    :param ip: The ip of the combiner (optional).
    :type ip: str
    :param config: The configuration of the combiner (optional).
    :type config: dict
    """

    def __init__(self, combiner_id, parent, name, address, fqdn, port, certificate=None, key=None, ip=None, config=None):
        """Initialize the combiner interface."""
        self.combiner_id = combiner_id
        self.parent = parent
        self.name = name
        self.address = address
        self.fqdn = fqdn
        self.port = port
        self.certificate = certificate
        self.key = key
        self.ip = ip

        if not config:
            self.config = {"max_clients": 8}
        else:
            self.config = config

    def get_certificate(self):
        """Get combiner certificate.

        :return: The combiner certificate.
        :rtype: str, None if no certificate is set.
        """
        if self.certificate:
            cert_b64 = base64.b64encode(self.certificate)
            return str(cert_b64).split("'")[1]
        else:
            return None

    def get_key(self):
        """Get combiner key.

        :return: The combiner key.
        :rtype: str, None if no key is set.
        """
        if self.key:
            key_b64 = base64.b64encode(self.key)
            return str(key_b64).split("'")[1]
        else:
            return None

    def flush_model_update_queue(self):
        """Reset the model update queue on the combiner."""
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)

        request = fedn.ControlRequest()

        try:
            control.FlushAggregationQueue(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError
            else:
                raise

    def set_aggregator(self, aggregator):
        """Set the active aggregator module.

        :param aggregator: The name of the aggregator module.
        :type config: str
        """
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)

        request = fedn.ControlRequest()
        p = request.parameter.add()
        p.key = "aggregator"
        p.value = aggregator

        try:
            control.SetAggregator(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError
            else:
                raise

    def set_server_functions(self, server_functions):
        """Set the function provider module.

        :param function provider: Stringified function provider code.
        :type config: str
        """
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)

        request = fedn.ControlRequest()
        p = request.parameter.add()
        p.key = "server_functions"
        p.value = server_functions

        try:
            control.SetServerFunctions(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError
            else:
                raise

    def submit(self, command: fedn.Command, parameters: Dict = None) -> fedn.ControlResponse:
        """Send a command to the combiner.

        :param command: The command to send.
        :type command: :class:`fedn.network.grpc.fedn_pb2.Command`
        :param parameters: The parameters for the command (optional).
        :type parameters: dict
        :return: The response from the combiner.
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ControlResponse`
        """
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)

        request = fedn.CommandRequest()

        request.command = command

        if parameters:
            request.parameters = json.dumps(parameters)

        try:
            response = control.SendCommand(request)
            return response
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError(f"Combiner {self.name} unavailable: {e}")
            else:
                raise

    def allowing_clients(self):
        """Check if the combiner is allowing additional client connections.

        :return: True if accepting, else False.
        :rtype: bool
        """
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        connector = rpc.ConnectorStub(channel)
        request = fedn.ConnectionRequest()

        try:
            response = connector.AcceptingClients(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError
            else:
                raise
        if response.status == fedn.ConnectionStatus.NOT_ACCEPTING:
            return False
        if response.status == fedn.ConnectionStatus.ACCEPTING:
            return True
        if response.status == fedn.ConnectionStatus.TRY_AGAIN_LATER:
            return False

        return False

    def list_active_clients(self, queue=1, max_retries=3, retry_delay=1.0):
        """List active clients with retry logic.

        :param queue: The channel (queue) to use (optional). Default is 1 = MODEL_UPDATE_REQUESTS channel.
        :type queue: int
        :param max_retries: How many times to retry if gRPC returns UNAVAILABLE.
        :type max_retries: int
        :param retry_delay: Seconds to wait before retrying.
        :type retry_delay: float
        :return: A list of active clients.
        :rtype: list
        """
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ConnectorStub(channel)
        request = fedn.ListClientsRequest()
        request.channel = queue
        try:
            response = control.ListActiveClients(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError
            else:
                raise
        return response.client


class ControlInterface:
    def __init__(self, address, port, certificate=None):
        """Initialize the control interface."""
        self.address = address
        self.port = port
        self.certificate = certificate

    def send_command(self, command: fedn.Command, command_type: str = None, parameters: Dict = None) -> fedn.ControlRequest:
        """Send a command to the control interface.

        :param command_type: The type of command to send.
        :type command_type: str
        :param parameters: The parameters for the command.
        :type parameters: dict
        :return: The response from the control interface.
        :rtype: dict
        """
        logger.info(f"Sending command {command} of type {command_type} to controller")
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)

        request = fedn.CommandRequest()
        request.command = command

        if command_type:
            request.command_type = command_type

        if parameters:
            request.parameters = json.dumps(parameters)

        try:
            response = control.SendCommand(request)
            return response
        except grpc.RpcError as e:
            raise CombinerUnavailableError(f"Control interface unavailable: {e}")

    def get_state(self) -> ControllerState:
        """Get the current state of the control interface.

        :return: The current state.
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ControlState`
        """
        logger.info(f"Getting control state from {self.address}:{self.port}")
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)

        request = fedn.ControlRequest()

        try:
            response = control.GetState(request)
            logger.info(f"Control state response: {response.state}")
            return ControllerState[response.state]

        except grpc.RpcError as e:
            raise CombinerUnavailableError(f"Control interface unavailable: {e}")
