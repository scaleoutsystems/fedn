import os
import tempfile
from io import BytesIO

import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
from fedn.network.storage.models.tempmodelstorage import TempModelStorage

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

    def get_tmp_path(self):
        """ Return a temporary output path compatible with save_model, load_model. """
        fd, path = tempfile.mkstemp()
        os.close(fd)
        return path

    def load_model_from_BytesIO(self, model_bytesio, helper):
        """ Load a model from a BytesIO object.

        :param model_bytesio: A BytesIO object containing the model.
        :type model_bytesio: :class:`io.BytesIO`
        :param helper: The helper object for the model.
        :type helper: :class:`fedn.utils.helperbase.HelperBase`
        :return: The model object.
        :rtype: return type of helper.load
        """
        path = self.get_tmp_path()
        with open(path, 'wb') as fh:
            fh.write(model_bytesio)
            fh.flush()
        model = helper.load(path)
        os.unlink(path)
        return model

    def serialize_model_to_BytesIO(self, model, helper):
        """ Serialize a model to a BytesIO object.

        :param model: The model object.
        :type model: return type of helper.load
        :param helper: The helper object for the model.
        :type helper: :class:`fedn.utils.helperbase.HelperBase`
        :return: A BytesIO object containing the model.
        :rtype: :class:`io.BytesIO`
        """
        outfile_name = helper.save(model)

        a = BytesIO()
        a.seek(0, 0)
        with open(outfile_name, 'rb') as f:
            a.write(f.read())
        os.unlink(outfile_name)
        return a

    def get_model(self, id):
        """ Download model with id 'id' from server.

        :param id: The model id.
        :type id: str
        :return: A BytesIO object containing the model.
        :rtype: :class:`io.BytesIO`, None if model does not exist.
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
        :type model: :class:`io.BytesIO`
        :param id: The model id.
        :type id: str
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
        """ RPC endpoints for uploading a model.

        :param request_iterator: The model request iterator.
        :type request_iterator: :class:`fedn.common.net.grpc.fedn_pb2.ModelRequest`
        :param context: The context object (unused)
        :type context: :class:`grpc._server._Context`
        :return: A model response object.
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.ModelResponse`
        """

        result = None
        for request in request_iterator:
            if request.status == fedn.ModelStatus.IN_PROGRESS:
                self.models.get_ptr(request.id).write(request.data)
                self.models.set_model_metadata(request.id, fedn.ModelStatus.IN_PROGRESS)

            if request.status == fedn.ModelStatus.OK and not request.data:
                result = fedn.ModelResponse(id=request.id, status=fedn.ModelStatus.OK,
                                            message="Got model successfully.")
                # self.models_metadata.update({request.id: fedn.ModelStatus.OK})
                self.models.set_model_metadata(request.id, fedn.ModelStatus.OK)
                self.models.get_ptr(request.id).flush()
                self.models.get_ptr(request.id).close()
                return result

    def Download(self, request, context):
        """ RPC endpoints for downloading a model.

        :param request: The model request object.
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.ModelRequest`
        :param context: The context object (unused)
        :type context: :class:`grpc._server._Context`
        :return: A model response iterator.
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.ModelResponse`
        """
        try:
            if self.models.get_model_metadata(request.id) != fedn.ModelStatus.OK:
                print("Error file is not ready", flush=True)
                yield fedn.ModelResponse(id=request.id, data=None, status=fedn.ModelStatus.FAILED)
        except Exception:
            print("Error file does not exist: {}".format(request.id), flush=True)
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
            print("Downloading went wrong: {} {}".format(request.id, e), flush=True)
