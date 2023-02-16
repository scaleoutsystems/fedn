import base64
import copy
import json
from io import BytesIO

import grpc

import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc


class CombinerUnavailableError(Exception):
    pass


class Channel:
    """ Wrapper for a gRPC channel. """

    def __init__(self, address, port, certificate=None):
        """ Create a channel.

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
            credentials = grpc.ssl_channel_credentials(
                root_certificates=copy.deepcopy(certificate))
            self.channel = grpc.secure_channel('{}:{}'.format(
                self.address, str(self.port)), credentials)
        else:
            self.channel = grpc.insecure_channel(
                '{}:{}'.format(self.address, str(self.port)))

    def get_channel(self):
        """ Get a channel.

        :return: An instance of a gRPC channel
        """
        return copy.copy(self.channel)


class CombinerInterface:
    """

    """

    def __init__(self, parent, name, address, fqdn, port, certificate=None, key=None, ip=None, config=None):
        self.parent = parent
        self.name = name
        self.address = address
        self.fqdn = fqdn
        self.port = port
        self.certificate = certificate
        self.key = key
        self.ip = ip

        if not config:
            self.config = {
                'max_clients': 8
            }
        else:
            self.config = config

    @classmethod
    def from_json(combiner_config):
        """ Initialize the combiner config from a json document.

        : return:
        """
        return CombinerInterface(**combiner_config)

    def to_dict(self):
        """ Export combiner configuration to a dictionary.

        : return:
        """

        data = {
            'parent': self.parent.to_dict(),
            'name': self.name,
            'address': self.address,
            'fqdn': self.fqdn,
            'port': self.port,
            'ip': self.ip,
            'certificate': None,
            'key': None
        }

        if self.certificate:
            cert_b64 = base64.b64encode(self.certificate)
            key_b64 = base64.b64encode(self.key)
            data['certificate'] = str(cert_b64).split('\'')[1]
            data['key'] = str(key_b64).split('\'')[1]

        try:
            data['report'] = self.report()
        except CombinerUnavailableError:
            data['report'] = None

        return data

    def to_json(self):
        """ Export combiner configuration to json.

        : return:
        """
        return json.dumps(self.to_dict())

    def get_certificate(self):
        """ Get combiner certificate.

        : return:
        """
        if self.certificate:
            cert_b64 = base64.b64encode(self.certificate)
            return str(cert_b64).split('\'')[1]
        else:
            return None

    def get_key(self):
        """ Get combiner key.

        : return:
        """
        if self.key:
            key_b64 = base64.b64encode(self.key)
            return str(key_b64).split('\'')[1]
        else:
            return None

    def report(self, config=None):
        """ Recieve a status report from the combiner. 

        : param config:
        : return:
        """
        channel = Channel(self.address, self.port,
                          self.certificate).get_channel()
        control = rpc.ControlStub(channel)
        request = fedn.ControlRequest()
        try:
            response = control.Report(request)
            data = {}
            for p in response.parameter:
                data[p.key] = p.value
            return data
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError
            else:
                raise

    def configure(self, config=None):
        """ Set configurations for the combiner.

        : param config: 
        """
        if not config:
            config = self.config
        channel = Channel(self.address, self.port,
                          self.certificate).get_channel()
        control = rpc.ControlStub(channel)

        request = fedn.ControlRequest()
        for key, value in config.items():
            p = request.parameter.add()
            p.key = key
            p.value = str(value)

        try:
            control.Configure(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError
            else:
                raise

    def start(self, config):
        """ Submit a compute plan to the combiner.

        : param config:
        : return:
        """
        channel = Channel(self.address, self.port,
                          self.certificate).get_channel()
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

        print("Response from combiner {}".format(response.message))
        return response

    def set_model_id(self, model_id):
        """ Set the current model_id at the combiner.

        : param model_id:
        """
        channel = Channel(self.address, self.port,
                          self.certificate).get_channel()
        control = rpc.ControlStub(channel)
        request = fedn.ControlRequest()
        p = request.parameter.add()
        p.key = 'model_id'
        p.value = str(model_id)

        try:
            control.Configure(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError
            else:
                raise

    def get_model(self, id):
        """ Retrive a model object from a combiner. """

        channel = Channel(self.address, self.port,
                          self.certificate).get_channel()
        modelservice = rpc.ModelServiceStub(channel)

        data = BytesIO()
        data.seek(0, 0)

        parts = modelservice.Download(fedn.ModelRequest(id=id))
        for part in parts:
            if part.status == fedn.ModelStatus.IN_PROGRESS:
                data.write(part.data)
            if part.status == fedn.ModelStatus.OK:
                return data
            if part.status == fedn.ModelStatus.FAILED:
                return None

    def allowing_clients(self):
        """ Check if the combiner is allowing additional client connections.

        : return:
        """
        channel = Channel(self.address, self.port,
                          self.certificate).get_channel()
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
