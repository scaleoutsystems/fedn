import base64
import copy
import json
import time
from io import BytesIO

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.network.combiner.roundhandler import RoundConfig


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

    def __init__(self, parent, name, address, fqdn, port, certificate=None, key=None, ip=None, config=None):
        """Initialize the combiner interface."""
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

    @classmethod
    def from_json(combiner_config):
        """Initialize the combiner config from a json document.

        :parameter combiner_config: The combiner configuration.
        :type combiner_config: dict
        :return: An instance of the combiner interface.
        :rtype: :class:`fedn.network.combiner.interfaces.CombinerInterface`
        """
        return CombinerInterface(**combiner_config)

    def to_dict(self):
        """Export combiner configuration to a dictionary.

        :return: A dictionary with the combiner configuration.
        :rtype: dict
        """
        data = {
            "parent": self.parent,
            "name": self.name,
            "address": self.address,
            "fqdn": self.fqdn,
            "port": self.port,
            "ip": self.ip,
            "certificate": None,
            "key": None,
            "config": self.config,
        }
        return data

    def to_json(self):
        """Export combiner configuration to json.

        :return: A json document with the combiner configuration.
        :rtype: str
        """
        return json.dumps(self.to_dict())

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

    def submit(self, config: RoundConfig):
        """Submit a compute plan to the combiner.

        :param config: The job configuration.
        :type config: dict
        :return: Server ControlResponse object.
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ControlResponse`
        """
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)
        request = fedn.ControlRequest()
        request.command = fedn.Command.START
        for k, v in config.items():
            p = request.parameter.add()
            p.key = str(k)
            p.value = str(v)

        try:
            response = control.Start(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError
            else:
                raise

        return response

    def get_model(self, id, timeout=10):
        """Download a model from the combiner server.

        :param id: The model id.
        :type id: str
        :return: A file-like object containing the model.
        :rtype: :class:`io.BytesIO`, None if the model is not available.
        """
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        modelservice = rpc.ModelServiceStub(channel)

        data = BytesIO()
        data.seek(0, 0)

        time_start = time.time()

        request = fedn.ModelRequest(id=id)
        request.sender.name = self.name
        request.sender.role = fedn.WORKER

        parts = modelservice.Download(request)
        for part in parts:
            if part.status == fedn.ModelStatus.IN_PROGRESS:
                data.write(part.data)
            if part.status == fedn.ModelStatus.OK:
                return data
            if part.status == fedn.ModelStatus.FAILED:
                return None
            if part.status == fedn.ModelStatus.UNKNOWN:
                if time.time() - time_start > timeout:
                    return None
                continue

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

    def list_active_clients(self, queue=1):
        """List active clients.

        :param queue: The channel (queue) to use (optional). Default is 1 = MODEL_UPDATE_REQUESTS channel.
            see :class:`fedn.network.grpc.fedn_pb2.Channel`
        :type channel: int
        :return: A list of active clients.
        :rtype: json
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
