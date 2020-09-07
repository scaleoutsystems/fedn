
import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
import grpc

class CombinerInterface:
    def __init__(self, parent, name, address, port):
        self.parent = parent
        self.name = name
        self.address = address
        self.port = port

    def start(self, config):
        channel = grpc.insecure_channel(self.address + ":" + str(self.port))
        control = rpc.ControlStub(channel)
        request = fedn.ControlRequest()
        request.command = fedn.Command.START
        for k, v in config.items():
            p = request.parameter.add()
            p.key = str(k)
            p.value = str(v)

        response = control.Start(request)
        print("Response from combiner {}".format(response.message))

    def set_model_id(self,model_id):        
        channel = grpc.insecure_channel(self.address + ":" + str(self.port))
        control = rpc.ControlStub(channel)
        request = fedn.ControlRequest()
        p = request.parameter.add()
        p.key = 'model_id'
        p.value = str(model_id)
        response = control.Configure(request)
        #return response.message

    def get_model_id(self):
        channel = grpc.insecure_channel(self.address + ":" + str(self.port))
        reducer = rpc.ReducerStub(channel)
        request = fedn.GetGlobalModelRequest()
        response = reducer.GetGlobalModel(request)
        return response.model_id

    def allowing_clients(self):
        print("Sending message to combiner", flush=True)
        channel = grpc.insecure_channel(self.address + ":" + str(self.port))
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

