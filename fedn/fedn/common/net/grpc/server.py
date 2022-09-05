from concurrent import futures

import grpc

import fedn.common.net.grpc.fedn_pb2_grpc as rpc


class Server:
    """

    """

    def __init__(self, servicer, modelservicer, config):

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=350))
        self.certificate = None

        if isinstance(servicer, rpc.CombinerServicer):
            rpc.add_CombinerServicer_to_server(servicer, self.server)
        if isinstance(servicer, rpc.ConnectorServicer):
            rpc.add_ConnectorServicer_to_server(servicer, self.server)
        if isinstance(servicer, rpc.ReducerServicer):
            rpc.add_ReducerServicer_to_server(servicer, self.server)
        if isinstance(modelservicer, rpc.ModelServiceServicer):
            rpc.add_ModelServiceServicer_to_server(modelservicer, self.server)
        if isinstance(servicer, rpc.CombinerServicer):
            rpc.add_ControlServicer_to_server(servicer, self.server)

        if config['secure']:
            print(f"Creating secure gRPCS server using certificate: {config['certificate']}", flush=True)
            server_credentials = grpc.ssl_server_credentials(
                ((config['key'], config['certificate'],),))
            self.server.add_secure_port(
                '[::]:' + str(config['port']), server_credentials)
        else:
            print("Creating insecure gRPC server", flush=True)
            self.server.add_insecure_port('[::]:' + str(config['port']))

    def start(self):
        """

        """
        print("Server started", flush=True)
        self.server.start()

    def stop(self):
        """

        """
        self.server.stop(0)
