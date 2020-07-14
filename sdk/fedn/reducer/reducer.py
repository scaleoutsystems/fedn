import os

import fedn.proto.alliance_pb2 as alliance
import fedn.proto.alliance_pb2_grpc as rpc
import grpc
from scaleout.repository.helpers import get_repository
from fedn.combiner.role import Role


class Reducer:

    def __init__(self, project):

        self.project = project
        self.global_model = None
        self.combiners = {}
        self.id = "reducer"
        try:
            self.id = self.id + str(os.environ['CLIENT_NAME'])
        except KeyError:
            pass

        # TODO remove enforced name
        self.id = "reducer"

        try:
            for unpack in self.project.config['Alliance']:
                # unpack = self.project.config['Alliance']
                address = unpack['controller_host']
                port = unpack['controller_port']
                channel = grpc.insecure_channel(address + ":" + str(port))
                connection = rpc.ReducerStub(channel)
                repository = get_repository(config=unpack['Repository'])
                bucket_name = unpack["Repository"]["minio_bucket"]
                from fedn.combiner.helpers import get_combiner
                combiner = get_combiner(project)(address, port, self.id, Role.COMBINER, repository)
                self.combiners.update(
                    {address: {'connection': connection,
                               'combiner': combiner,
                               'repository': repository,
                               'bucket': bucket_name}})

                print("REDUCER: : {} connected to {}:{}".format(self.id, address, port), flush=True)

        except KeyError as e:
            print("REDUCER: could not get all values from config file {}".format(e))

    def request_model(self):
        request = alliance.GetGlobalModelRequest()
        request.sender.name = self.id
        request.sender.role = alliance.REDUCER

        response = self.connection.GetGlobalModel(request)
        return response

    def run(self):
        import time
        print("REDUCER: starting reducer", flush=True)
        time.sleep(20)
        print("REDUCER: activating reducer", flush=True)

        inc = 0
        while True:
            print("REDUCER: running.", flush=True)
            time.sleep(1)
            inc += 1
            if inc > 10:
                print("REDUCER: requesting model!", flush=True)
                msg = self.request_model()
                self.combiner.receive_model_candidate(msg.model_id)
                print("got model! {}".format(msg.model_id), flush=True)
                inc = 0

            current_nr_models = self.combiner.model_updates.qsize()
            if current_nr_models > 1:
                model = self.combiner.combine_models(current_nr_models)
                import tempfile
                fod, outfile_name = tempfile.mkstemp(suffix='.h5')
                model.save(outfile_name)
                self.global_model = self.repository.set_model(outfile_name, is_file=True)
                import sys
                print("NEW reduced combined global model! {}".format(self.global_model), flush=True)
