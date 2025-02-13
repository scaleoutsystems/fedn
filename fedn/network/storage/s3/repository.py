"""Interface for storing model objects and compute packages in S3 compatible storage."""

import datetime
import os
import uuid
from typing import Union

from fedn.common.log_config import logger
from fedn.network.storage.s3.miniorepository import MINIORepository
from fedn.network.storage.s3.saasrepository import SAASRepository


class Repository:
    """Interface for storing model objects and compute packages in S3 compatible storage."""

    def __init__(self, config: dict, init_buckets: bool = True, storage_type: str = "MINIO") -> None:
        """Initialize the repository.

        :param config: Configuration dictionary for credentials and bucket names.
        :type config: dict
        :param init_buckets: Whether to initialize buckets, defaults to True
        :type init_buckets: bool, optional
        :param storage_type: Type of storage to use, defaults to "MINIO"
        :type storage_type: str, optional
        """
        self.model_bucket = os.environ.get("FEDN_MODEL_BUCKET", config["storage_bucket"])
        self.context_bucket = os.environ.get("FEDN_CONTEXT_BUCKET", config["context_bucket"])
        try:
            self.prediction_bucket = os.environ.get("FEDN_PREDICTION_BUCKET", config["prediction_bucket"])
        except KeyError:
            self.prediction_bucket = "fedn-prediction"

        # TODO: Make a plug-in solution
        storage_type = os.environ.get("FEDN_STORAGE_TYPE", storage_type)
        if storage_type == "SAAS":
            self.client = SAASRepository(config)
        elif storage_type == "MINIO":
            self.client = MINIORepository(config)
        else:
            # Default to MinIO.
            self.client = MINIORepository(config)

        if init_buckets:
            self.client.create_bucket(self.context_bucket)
            self.client.create_bucket(self.model_bucket)
            self.client.create_bucket(self.prediction_bucket)

    def get_model(self, model_id: str) -> bytes:
        """Retrieve a model with id model_id.

        :param model_id: Unique identifier for model to retrieve.
        :type model_id: str
        :return: The model object
        :rtype: bytes
        """
        logger.info("Client {} trying to get model with id: {}".format(self.client.name, model_id))
        return self.client.get_artifact(model_id, self.model_bucket)

    def get_model_stream(self, model_id: str) -> bytes:
        """Retrieve a stream handle to model with id model_id.

        :param model_id: Unique identifier for model to retrieve.
        :type model_id: str
        :return: Handle to model object
        :rtype: bytes
        """
        logger.info("Client {} trying to get model with id: {}".format(self.client.name, model_id))
        return self.client.get_artifact_stream(model_id, self.model_bucket)

    def set_model(self, model: Union[bytes, str], is_file: bool = True) -> str:
        """Upload model object.

        :param model: The model object
        :type model: Union[bytes, str]
        :param is_file: True if model is a file name, else False
        :type is_file: bool, optional
        :return: id for the uploaded object
        :rtype: str
        """
        model_id = uuid.uuid4()

        try:
            self.client.set_artifact(str(model_id), model, bucket=self.model_bucket, is_file=is_file)
        except Exception as e:
            logger.error("Failed to upload model with ID {} to repository. Error: {}".format(model_id, e))
            raise Exception(f"Failed to upload model with ID {model_id} to repository.") from e
        return str(model_id)

    def delete_model(self, model_id: str) -> None:
        """Delete model.

        :param model_id: The id of the model to delete
        :type model_id: str
        """
        try:
            self.client.delete_artifact(model_id, bucket=self.model_bucket)
        except Exception as e:
            logger.error("Failed to delete model {} from repository. Error: {}".format(model_id, e))
            raise Exception(f"Failed to delete model {model_id} from repository.") from e

    def set_compute_package(self, name: str, compute_package: Union[bytes, str], is_file: bool = True) -> None:
        """Upload compute package.

        :param name: The name of the compute package.
        :type name: str
        :param compute_package: The compute package
        :type compute_package: Union[bytes, str]
        :param is_file: True if model is a file name, else False
        :type is_file: bool, optional
        """
        try:
            self.client.set_artifact(name, compute_package, bucket=self.context_bucket, is_file=is_file)
        except Exception as e:
            logger.error("Failed to write compute package to repository. Error: {}".format(e))
            raise Exception("Failed to write compute package to repository.") from e

    def get_compute_package(self, compute_package: str) -> bytes:
        """Retrieve compute package from object store.

        :param compute_package: The name of the compute package.
        :type compute_package: str
        :return: Compute package.
        :rtype: bytes
        """
        try:
            data = self.client.get_artifact(compute_package, bucket=self.context_bucket)
        except Exception as e:
            logger.error("Failed to get compute package from repository. Error: {}".format(e))
            raise Exception("Failed to get compute package from repository.") from e
        return data

    def delete_compute_package(self, compute_package: str) -> None:
        """Delete a compute package from storage.

        :param compute_package: The name of the compute package
        :type compute_package: str
        """
        try:
            self.client.delete_artifact(compute_package, bucket=self.context_bucket)
        except Exception as e:
            logger.error("Failed to delete compute package from repository. Error: {}".format(e))
            raise Exception("Failed to delete compute package from repository.") from e

    def presigned_put_url(self, bucket: str, object_name: str, expires: datetime.timedelta = datetime.timedelta(hours=1)) -> str:
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
