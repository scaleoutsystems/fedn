import os
import tempfile
import threading
from io import BytesIO
from typing import Generator

import grpc

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger
from fedn.network.storage.models.tempmodelstorage import TempModelStorage
from fedn.network.storage.s3.repository import Repository
from fedn.utils.model import FednModel

CHUNK_SIZE = 1 * 1024 * 1024


def upload_request_generator(model_stream: BytesIO):
    """Generator function for model upload requests for the client

    :param mdl: The model update object.
    :type mdl: BytesIO
    :return: A model update request.
    :rtype: fedn.FileChunk
    """
    while True:
        b = model_stream.read(CHUNK_SIZE)
        if b:
            yield fedn.FileChunk(data=b)
        else:
            break


def bytesIO_request_generator(mdl, request_function, args):
    """Generator function for model upload requests.

    :param mdl: The model update object.
    :type mdl: BytesIO
    :param request_function: Function for sending requests.
    :type request_function: Function
    :param args: request arguments, excluding data argument.
    :type args: dict
    :return: Yields grpc request for streaming.
    :rtype: grpc request generator.
    """
    while True:
        b = mdl.read(CHUNK_SIZE)
        if b:
            result = request_function(data=b, **args)
        else:
            result = request_function(data=None, **args)
        yield result
        if not b:
            break


def model_params_as_fednmodel(model_params, helper=None):
    if isinstance(model_params, list):
        return FednModel.from_model_params(model_params, helper=helper)
    elif not isinstance(model_params, FednModel):
        raise ValueError("model_params must be a list of numpy arrays or a FednModel instance.")
    return model_params


def unpack_model(request_iterator, helper):
    """Unpack an incoming model sent in chunks from a request iterator.

    :param request_iterator: A streaming iterator from an gRPC service.
    :return: The reconstructed model parameters.
    """
    try:
        model = FednModel.from_chunk_generator(request.data for request in request_iterator if request.data)
    except MemoryError as e:
        logger.error(f"Memory error occured when loading model, reach out to the FEDn team if you need a solution to this. {e}")
        raise
    except Exception as e:
        logger.error(f"Exception occured during model loading: {e}")
        raise

    return model.get_model_params(helper)


def get_tmp_path():
    """Return a temporary output path compatible with save_model, load_model."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    return path


def serialize_model_to_BytesIO(model, helper):
    """Serialize a model to a BytesIO object.

    :param model: The model object.
    :type model: return type of helper.load
    :param helper: The helper object for the model.
    :type helper: :class:`fedn.utils.helperbase.HelperBase`
    :return: A BytesIO object containing the model.
    :rtype: :class:`io.BytesIO`
    """
    return FednModel.from_model_params(model, helper=helper)


class ModelService(rpc.ModelServiceServicer):
    """Service for handling download and upload of models to the server."""

    def __init__(self, repository: Repository):
        """Initialize the temporary model storage."""
        self.temp_model_storage = TempModelStorage()
        self.repository = repository

    def exist(self, model_id):
        """Check if a model exists on the server.

        :param model_id: The model id.
        :return: True if the model exists, else False.
        """
        return self.temp_model_storage.exist(model_id)

    def get_model(self, model_id):
        """Get a model from the server.

        :param model_id: The model id.
        :return: The model object.
        :rtype: :class:`fedn.network.storage.models.tempmodelstorage.FednModel`
        """
        if not self.temp_model_storage.exist(model_id):
            logger.error(f"ModelServicer: Model {model_id} does not exist.")
            raise ValueError(f"Model {model_id} does not exist in temporary storage.")
        return self.temp_model_storage.get(model_id)

    def model_ready(self, model_id):
        """Check if a model is ready on the server.

        :param model_id: The model id.
        :return: True if the model is ready, else False.
        """
        return self.temp_model_storage.is_ready(model_id)

    def set_model(self, model, model_id):
        """Upload model to server.

        :param model: A model object (BytesIO)
        :type model: :class:`io.BytesIO`
        :param id: The model id.
        :type id: str
        """
        fedn_model = model_params_as_fednmodel(model)
        self.temp_model_storage.set_model(model_id, fedn_model)

    def fetch_model_from_repository(self, model_id, blocking: bool = False):
        """Fetch model from the repository and store it in the temporary model storage.

        :param model_id: The model id to fetch.
        :type model_id: str
        """
        logger.info(f"Fetching model {model_id} from repository.")
        try:
            model = self.repository.get_model_stream(model_id)
            if model:
                logger.info(f"Model {model_id} fetched and stored successfully.")
                if blocking:
                    return self.temp_model_storage.set_model_from_stream(model_id, model, auto_managed=True)
                else:
                    threading.Thread(target=lambda: self.temp_model_storage.set_model_from_stream(model_id, model, auto_managed=True)).start()
                    return True
            else:
                logger.error(f"Model {model_id} not found in repository.")
                return False
        except Exception as e:
            logger.error(f"Error fetching model {model_id} from repository: {e}")
            return False

    # Model Service
    def Upload(self, filechunk_iterator: Generator[fedn.FileChunk, None, None], context: grpc.ServicerContext):
        """RPC endpoints for uploading a model.

        :param filechunk_iterator: The model request iterator.
        :type filechunk_iterator: :class:`fedn.network.grpc.fedn_pb2.FileChunk`
        :param context: The context object
        :type context: :class:`grpc._server._Context`
        :return: A model response object.
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ModelResponse`
        """
        logger.debug("grpc.ModelService.Upload: Called")

        metadata = dict(context.invocation_metadata())
        model_id = metadata.get("model-id")
        checksum = metadata.get("checksum")

        if not model_id:
            logger.error("ModelServicer: Model ID not provided.")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model ID not provided.")

        model_chunks = (chunk.data for chunk in filechunk_iterator if chunk.data)
        result = self.temp_model_storage.set_model_with_generator(model_id, model_chunks, checksum)
        if result:
            return fedn.ModelResponse(status=fedn.ModelStatus.OK, message="Got model successfully.")
        else:
            return fedn.ModelResponse(status=fedn.ModelStatus.FAILED, message="Failed to upload model.")

    def Download(self, request: fedn.ModelRequest, context: grpc.ServicerContext):
        """RPC endpoints for downloading a model.

        :param request: The model request object.
        :type request: :class:`fedn.network.grpc.fedn_pb2.ModelRequest`
        :param context: The context object (unused)
        :type context: :class:`grpc._server._Context`
        :return: A model response iterator.
        :rtype: :class:`fedn.network.grpc.fedn_pb2.FileChunk`
        """
        logger.info(f"grpc.ModelService.Download: {request.sender.role}:{request.sender.client_id} requested model {request.model_id}")
        if not request.model_id:
            logger.error("ModelServicer: Model ID not provided.")
            context.abort(grpc.StatusCode.UNAVAILABLE, "Model ID not provided.")

        if not self.temp_model_storage.is_ready(request.model_id):
            if self.temp_model_storage.exist(request.model_id):
                logger.error(f"ModelServicer: Model file is not ready: {request.model_id}")
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model file is not ready.")
            else:
                logger.error(f"ModelServicer: Model file does not exist: {request.model_id}. Trying to start automatic caching")
                file_is_downloading = self.fetch_model_from_repository(request.model_id)
                if not file_is_downloading:
                    logger.error(f"ModelServicer: Model file does not exist: {request.model_id}.")
                    context.abort(grpc.StatusCode.UNAVAILABLE, "Model file does not exist. ")
                else:
                    logger.info(f"ModelServicer: Caching started: {request.model_id}.")
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model file is not ready. Starting automatic caching.")

        try:
            model: FednModel = self.temp_model_storage.get(request.model_id)
            stream = model.get_stream()
            while True:
                chunk = stream.read(CHUNK_SIZE)
                if chunk:
                    yield fedn.FileChunk(data=chunk)
                else:
                    break
        except Exception as e:
            logger.error("Downloading went wrong: {} {}".format(request.model_id, e))
            context.abort(grpc.StatusCode.UNKNOWN, "Download failed.")

        context.set_trailing_metadata((("checksum", self.temp_model_storage.get_checksum(request.model_id)),))
