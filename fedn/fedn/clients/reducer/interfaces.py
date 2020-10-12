import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
import grpc

class CombinerUnavailableError(Exception):
    pass

class Channel:
    def __init__(self, address, port, certificate):
        self.address = address
        self.port = port
        self.certificate = certificate
        #print("USING THIS CERTIFICATE: \n\n\n {} \n\n\n\n".format(certificate), flush=True)
        if self.certificate:
            import copy
            credentials = grpc.ssl_channel_credentials(root_certificates=copy.deepcopy(certificate))
            self.channel = grpc.secure_channel('{}:{}'.format(self.address, str(self.port)), credentials)
        else:
            self.channel = grpc.insecure_channel('{}:{}'.format(self.address, str(self.port)))

    def get_channel(self):
        import copy
        return copy.copy(self.channel)


class CombinerInterface:
    def __init__(self, parent, name, address, port, certificate=None, key=None, config=None):
        self.parent = parent
        self.name = name
        self.address = address
        self.port = port
        self.certificate = certificate
        self.key = key

        if not config:
            self.config = {
                'max_clients': 8
            }
        else:
            self.config = config

    def report(self,config=None):
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)
        request = fedn.ControlRequest()
        try:
            response = control.Report(request)
            data = {}
            for p in response.parameter:
                data[p.key] = p.value
            return data
        # TODO: Handle error more specifically 
        except Exception as e:
            print("Combiner failed to report: {}".format(e),flush=True)
            return None


    def configure(self,config=None):
        if not config:
            config = self.config
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)
        request = fedn.ControlRequest()
        for key,value in config.items():  
            p = request.parameter.add()
            p.key = key
            p.value = str(value)
        response = control.Configure(request)


    def start(self, config):
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)
        request = fedn.ControlRequest()
        request.command = fedn.Command.START
        for k, v in config.items():
            p = request.parameter.add()
            p.key = str(k)
            p.value = str(v)

        response = control.Start(request)
        print("Response from combiner {}".format(response.message))
        return response

    def set_model_id(self, model_id):
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        control = rpc.ControlStub(channel)
        request = fedn.ControlRequest()
        p = request.parameter.add()
        p.key = 'model_id'
        p.value = str(model_id)
        response = control.Configure(request)
        # return response.message

    def get_model_id(self):
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        reducer = rpc.ReducerStub(channel)
        request = fedn.GetGlobalModelRequest()
        try:
            response = reducer.GetGlobalModel(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise CombinerUnavailableError
            else:
                raise

        return response.model_id

    def get_model(self, id=None):
        """ Retrive the model bundle from a combiner. """

        channel = Channel(self.address, self.port, self.certificate).get_channel()
        modelservice = rpc.ModelServiceStub(channel)

        if not id: 
            id = self.get_model_id()

        from io import BytesIO
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
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        connector = rpc.ConnectorStub(channel)
        request = fedn.ConnectionRequest()
        response = connector.AcceptingClients(request)
        if response.status == fedn.ConnectionStatus.NOT_ACCEPTING:
            return False
        if response.status == fedn.ConnectionStatus.ACCEPTING:
            return True
        if response.status == fedn.ConnectionStatus.TRY_AGAIN_LATER:
            return False

        return False


class ReducerInferenceInterface:
    def __init__(self):
        self.model_wrapper = None

    def set(self, model):
        self.model_wrapper = model

    def infer(self, params):
        results = None
        if self.model_wrapper:
            results = self.model_wrapper.infer(params)

        return results
