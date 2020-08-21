
import fedn.proto.alliance_pb2 as alliance
import fedn.proto.alliance_pb2_grpc as rpc
import grpc

class Reducer:
    def __init__(self,config):

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

        # TODO setup sevices according to execution context!
        #rpc.add_CombinerServicer_to_server(self, self.server)
        #rpc.add_ConnectorServicer_to_server(self, self.server)
        rpc.add_ReducerServicer_to_server(self, self.server)
        #rpc.add_ModelServiceServicer_to_server(self, self.server)

        self.server.add_insecure_port('[::]:' + str(port))

        # Actual FedML algorithm
        job_config, _ = self.controller.get_config()
        self.orchestrator = get_orchestrator(job_config)(address, port, self.id, self.role, self.repository)

        self.server.start()

    def run(self):

        # 1. Setup package

        # 2. Notify Clients

        # 3. Ready for start

        # 4. Autostart (or start by interface)

        pass
