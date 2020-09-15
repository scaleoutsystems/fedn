import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
import grpc

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
    def __init__(self, parent, name, address, port, certificate=None, key=None):
        self.parent = parent
        self.name = name
        self.address = address
        self.port = port
        self.certificate = certificate
        self.key = key

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
        response = reducer.GetGlobalModel(request)
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
        #import time
        #import random
        #time.sleep(10.0 * random.random() / 2.0)  # try to debug concurrency issues? wait at most 5 before downloading

        parts = modelservice.Download(fedn.ModelRequest(id=id))
        for part in parts:
            if part.status == fedn.ModelStatus.IN_PROGRESS:
                data.write(part.data)
            if part.status == fedn.ModelStatus.OK:
                return data
            if part.status == fedn.ModelStatus.FAILED:
                return None

    def allowing_clients(self):
        print("Sending message to combiner", flush=True)
        channel = Channel(self.address, self.port, self.certificate).get_channel()
        connector = rpc.ConnectorStub(channel)
        request = fedn.ConnectionRequest()
        response = connector.AcceptingClients(request)
        if response.status == fedn.ConnectionStatus.NOT_ACCEPTING:
            print("Sending message to combiner 2", flush=True)
            return False
        if response.status == fedn.ConnectionStatus.ACCEPTING:
            print("Sending message to combiner 3", flush=True)
            return True
        if response.status == fedn.ConnectionStatus.TRY_AGAIN_LATER:
            print("Sending message to combiner 4", flush=True)
            return False

        print("Sending message to combiner 5??", flush=True)
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
