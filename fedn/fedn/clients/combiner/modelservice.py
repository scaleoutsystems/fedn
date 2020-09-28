import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
from fedn.common.storage.models.tempmodelstorage import TempModelStorage

CHUNK_SIZE = 1024 * 1024


class ModelService(rpc.ModelServiceServicer):

    def __init__(self):
        self.models = TempModelStorage()
        # self.models = defaultdict(io.BytesIO)
        # self.models_metadata = {}

    def exist(self, model_id):
        return self.models.exist(model_id)

    def get_model(self, id):
        from io import BytesIO
        data = BytesIO()
        data.seek(0, 0)
        import time
        import random
 #       time.sleep(10.0 * random.random() / 2.0)  # try to debug concurrency issues? wait at most 5 before downloading
        # print("REACHED DOWNLOAD Trying now with id {}".format(id), flush=True)

        # print("TRYING DOWNLOAD 1.", flush=True)
        parts = self.Download(fedn.ModelRequest(id=id), self)
        for part in parts:
            # print("TRYING DOWNLOAD 2.", flush=True)
            if part.status == fedn.ModelStatus.IN_PROGRESS:
                # print("WRITING PART FOR MODEL:{}".format(id), flush=True)
                data.write(part.data)

            if part.status == fedn.ModelStatus.OK:
                # print("DONE WRITING MODEL RETURNING {}".format(id), flush=True)
                # self.lock.release()
                return data
            if part.status == fedn.ModelStatus.FAILED:
                # print("FAILED TO DOWNLOAD MODEL::: bailing!", flush=True)
                return None

    def set_model(self, model, id):
        from io import BytesIO

        if not isinstance(model, BytesIO):
            bt = BytesIO()

            written_total = 0
            for d in model.stream(32 * 1024):
                written = bt.write(d)
                written_total += written
        else:
            bt = model

        bt.seek(0, 0)

        def upload_request_generator(mdl):
            while True:
                b = mdl.read(CHUNK_SIZE)
                if b:
                    result = fedn.ModelRequest(data=b, id=id, status=fedn.ModelStatus.IN_PROGRESS)
                else:
                    result = fedn.ModelRequest(id=id, data=None, status=fedn.ModelStatus.OK)
                yield result
                if not b:
                    break

        result = self.Upload(upload_request_generator(bt), self)

    ## Model Service
    def Upload(self, request_iterator, context):
        # print("STARTING UPLOAD!", flush=True)
        result = None
        for request in request_iterator:
            if request.status == fedn.ModelStatus.IN_PROGRESS:
                self.models.get_ptr(request.id).write(request.data)
                self.models.set_meta(request.id, fedn.ModelStatus.IN_PROGRESS)

            if request.status == fedn.ModelStatus.OK and not request.data:
                result = fedn.ModelResponse(id=request.id, status=fedn.ModelStatus.OK,
                                            message="Got model successfully.")
                # self.models_metadata.update({request.id: fedn.ModelStatus.OK})
                self.models.set_meta(request.id, fedn.ModelStatus.OK)
                self.models.get_ptr(request.id).flush()
                self.models.get_ptr(request.id).close()
                return result

    def Download(self, request, context):

        try:
            if self.models.get_meta(request.id) != fedn.ModelStatus.OK:
                print("Error file is not ready", flush=True)
                yield fedn.ModelResponse(id=request.id, data=None, status=fedn.ModelStatus.FAILED)
        except Exception as e:
            print("Error file does not exist", flush=True)
            yield fedn.ModelResponse(id=request.id, data=None, status=fedn.ModelStatus.FAILED)

        try:
            obj = self.models.get(request.id)
            import sys
            with obj as f:
                while True:
                    piece = f.read(CHUNK_SIZE)
                    if len(piece) == 0:
                        yield fedn.ModelResponse(id=request.id, data=None, status=fedn.ModelStatus.OK)
                        return
                    yield fedn.ModelResponse(id=request.id, data=piece, status=fedn.ModelStatus.IN_PROGRESS)
        except Exception as e:
            print("Downloading went wrong! {}".format(e), flush=True)
