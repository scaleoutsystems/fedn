import datetime
import uuid

from fedn.common.log_config import logger
from fedn.network.storage.s3.miniorepository import MINIORepository


class Repository:
    """Interface for storing model objects and compute packages in S3 compatible storage."""

    def __init__(self, config, init_buckets=True):
        self.model_bucket = config["storage_bucket"]
        self.context_bucket = config["context_bucket"]
        try:
            self.inference_bucket = config["inference_bucket"]
        except KeyError:
            self.inference_bucket = "fedn-inference"

        # TODO: Make a plug-in solution
        self.client = MINIORepository(config)

        if init_buckets:
            self.client.create_bucket(self.context_bucket)
            self.client.create_bucket(self.model_bucket)
            self.client.create_bucket(self.inference_bucket)

    def get_model(self, model_id):
        """Retrieve a model with id model_id.

        :param model_id: Unique identifier for model to retrive.
        :return: The model object
        """
        logger.info("Client {} trying to get model with id: {}".format(self.client.name, model_id))
        return self.client.get_artifact(model_id, self.model_bucket)

    def get_model_stream(self, model_id):
        """Retrieve a stream handle to model with id model_id.

        :param model_id:
        :return: Handle to model object
        """
        logger.info("Client {} trying to get model with id: {}".format(self.client.name, model_id))
        return self.client.get_artifact_stream(model_id, self.model_bucket)

    def set_model(self, model, is_file=True):
        """Upload model object.

        :param model: The model object
        :type model: BytesIO or str file name.
        :param is_file: True if model is a file name, else False
        :return: id for the uploaded object (str)
        """
        model_id = uuid.uuid4()

        try:
            self.client.set_artifact(str(model_id), model, bucket=self.model_bucket, is_file=is_file)
        except Exception:
            logger.error("Failed to upload model with ID {} to repository.".format(model_id))
            raise
        return str(model_id)

    def delete_model(self, model_id):
        """Delete model.

        :param model_id: The id of the model to delete
        :type model_id: str
        """
        try:
            self.client.delete_artifact(model_id, bucket=self.model_bucket)
        except Exception:
            logger.error("Failed to delete model {} repository.".format(model_id))
            raise

    def set_compute_package(self, name, compute_package, is_file=True):
        """Upload compute package.

        :param name: The name of the compute package.
        :type name: str
        :param compute_package: The compute package
        :type compute_pacakge: BytesIO or str file name.
        :param is_file: True if model is a file name, else False
        """
        try:
            self.client.set_artifact(str(name), compute_package, bucket=self.context_bucket, is_file=is_file)
        except Exception:
            logger.error("Failed to write compute_package to repository.")
            raise

    def get_compute_package(self, compute_package):
        """Retrieve compute package from object store.

        :param compute_package: The name of the compute package.
        :type compute_pacakge: str
        :return: Compute package.
        """
        try:
            data = self.client.get_artifact(compute_package, bucket=self.context_bucket)
        except Exception:
            logger.error("Failed to get compute_package from repository.")
            raise
        return data

    def delete_compute_package(self, compute_package):
        """Delete a compute package from storage.

        :param compute_package: The name of the compute_package
        :type compute_package: str
        """
        try:
            self.client.delete_artifact(compute_package, bucket=[self.context_bucket])
        except Exception:
            logger.error("Failed to delete compute_package from repository.")
            raise

    def presigned_put_url(self, bucket: str, object_name: str, expires: datetime.timedelta = datetime.timedelta(hours=1)):
        """Generate a presigned URL for an upload object request.

        :param bucket: The bucket name
        :type bucket: str
        :param object_name: The object name
        :type object_name: str
        :param expires: The time the URL is valid
        :type expires: datetime.timedelta
        :return: The URL
        :rtype: str
        """
        return self.client.client.presigned_put_object(bucket, object_name, expires)

    def presigned_get_url(self, bucket: str, object_name: str, expires: datetime.timedelta = datetime.timedelta(hours=1)) -> str:
        """Generate a presigned URL for a download object request.

        :param bucket: The bucket name
        :type bucket: str
        :param object_name: The object name
        :type object_name: str
        :param expires: The time the URL is valid
        :type expires: datetime.timedelta
        :return: The URL
        :rtype: str
        """
        return self.client.client.presigned_get_object(bucket, object_name, expires)
