from io import BytesIO

import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
from fedn.common.storage.models.tempmodelstorage import TempModelStorage

CHUNK_SIZE = 1024 * 1024


class ModelService(rpc.ModelServiceServicer):
    """ Service for handling download and upload of models to the server.

    """

    def __init__(self):
        self.models = TempModelStorage()

    def exist(self, model_id):
        """ Check if a model exists on the server.

        :param model_id: The model id.
        :return: True if the model exists, else False.
        """
        return self.models.exist(model_id)

    def get_model(self, id):
        """ Download model with id 'id' from server.

        :param id:
        :return:
        """

        data = BytesIO()
        data.seek(0, 0)

        parts = self.Download(fedn.ModelRequest(id=id), self)
        for part in parts:
            if part.status == fedn.ModelStatus.IN_PROGRESS:
                data.write(part.data)

            if part.status == fedn.ModelStatus.OK:
                return data
            if part.status == fedn.ModelStatus.FAILED:
                return None

    def set_model(self, model, id):
        """ Upload model to server.

        :param model: A model object (BytesIO)
        :param id: The model id.
        """

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
            """

            :param mdl:
            """
            while True:
                b = mdl.read(CHUNK_SIZE)
                if b:
                    result = fedn.ModelRequest(
                        data=b, id=id, status=fedn.ModelStatus.IN_PROGRESS)
                else:
                    result = fedn.ModelRequest(
                        id=id, data=None, status=fedn.ModelStatus.OK)
                yield result
                if not b:
                    break

        # TODO: Check result
        _ = self.Upload(upload_request_generator(bt), self)

    # Model Service
    def Upload(self, request_iterator, context):
        """ RPC for uploading a model.

        :param request_iterator:
        :param context:
        :return:
        """

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
        """ RPC for downloading a model.

        :param request:
        :param context:
        :return:
        """
        try:
            if self.models.get_meta(request.id) != fedn.ModelStatus.OK:
                print("Error file is not ready", flush=True)
                yield fedn.ModelResponse(id=request.id, data=None, status=fedn.ModelStatus.FAILED)
        except Exception:
            print("Error file does not exist", flush=True)
            yield fedn.ModelResponse(id=request.id, data=None, status=fedn.ModelStatus.FAILED)

        try:
            obj = self.models.get(request.id)
            with obj as f:
                while True:
                    piece = f.read(CHUNK_SIZE)
                    if len(piece) == 0:
                        yield fedn.ModelResponse(id=request.id, data=None, status=fedn.ModelStatus.OK)
                        return
                    yield fedn.ModelResponse(id=request.id, data=piece, status=fedn.ModelStatus.IN_PROGRESS)
        except Exception as e:
            print("Downloading went wrong! {}".format(e), flush=True)
