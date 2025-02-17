"""Implementation of the Repository interface for SaaS deployment."""

import io
import os
from typing import IO, List

from minio import Minio
from minio.error import InvalidResponseError

from fedn.common.log_config import logger
from fedn.network.storage.s3.base import RepositoryBase


class SAASRepository(RepositoryBase):
    """Class implementing Repository for SaaS deployment."""

    client = None

    def __init__(self, config: dict) -> None:
        """Initialize object.

        :param config: Dictionary containing configuration for credentials and bucket names.
        :type config: dict
        """
        super().__init__()
        self.name = "SAASRepository"
        self.project_slug = os.environ.get("FEDN_JWT_CUSTOM_CLAIM_VALUE")

        # Check environment variables first. If they are not set, then use values from config file.
        access_key = os.environ.get("FEDN_ACCESS_KEY", config["storage_access_key"])
        secret_key = os.environ.get("FEDN_SECRET_KEY", config["storage_secret_key"])
        storage_hostname = os.environ.get("FEDN_STORAGE_HOSTNAME", config["storage_hostname"])
        storage_port = os.environ.get("FEDN_STORAGE_PORT", config["storage_port"])
        storage_secure_mode = os.environ.get("FEDN_STORAGE_SECURE_MODE", config["storage_secure_mode"])
        storage_region = os.environ.get("FEDN_STORAGE_REGION") or config.get("storage_region", "auto")

        storage_secure_mode = storage_secure_mode.lower() == "true"

        self.client = Minio(
            f"{storage_hostname}:{storage_port}",
            access_key=access_key,
            secret_key=secret_key,
            secure=storage_secure_mode,
            region=storage_region,
        )

    def set_artifact(self, instance_name: str, instance: IO, bucket: str, is_file: bool = False) -> bool:
        """Set object with name instance_name.

        :param instance_name: The name of the object
        :type instance_name: str
        :param instance: The object
        :type instance: Any
        :param bucket: The bucket name
        :type bucket: str
        :param is_file: Whether the instance is a file, defaults to False
        :type is_file: bool, optional
        :return: True if the artifact was set successfully
        :rtype: bool
        """
        instance_name = f"{self.project_slug}/{instance_name}"
        logger.info(f"Setting artifact: {instance_name} in bucket: {bucket}")

        try:
            if is_file:
                self.client.fput_object(bucket, instance_name, instance)
            else:
                self.client.put_object(bucket, instance_name, io.BytesIO(instance), len(instance))
        except Exception as e:
            logger.error(f"Failed to upload artifact: {instance_name} to bucket: {bucket}. Error: {e}")
            raise Exception(f"Could not load data into bytes: {e}") from e

        return True

    def get_artifact(self, instance_name: str, bucket: str) -> bytes:
        """Retrieve object with name instance_name.

        :param instance_name: The name of the object to retrieve
        :type instance_name: str
        :param bucket: The bucket name
        :type bucket: str
        :return: The retrieved object
        :rtype: bytes
        """
        instance_name = f"{self.project_slug}/{instance_name}"
        logger.info(f"Getting artifact: {instance_name} from bucket: {bucket}")

        try:
            data = self.client.get_object(bucket, instance_name)
            return data.read()
        except Exception as e:
            logger.error(f"Failed to fetch artifact: {instance_name} from bucket: {bucket}. Error: {e}")
            raise Exception(f"Could not fetch data from bucket: {e}") from e
        finally:
            data.close()
            data.release_conn()

    def get_artifact_stream(self, instance_name: str, bucket: str) -> io.BytesIO:
        """Return a stream handler for object with name instance_name.

        :param instance_name: The name of the object
        :type instance_name: str
        :param bucket: The bucket name
        :type bucket: str
        :return: Stream handler for object instance_name
        :rtype: io.BytesIO
        """
        instance_name = f"{self.project_slug}/{instance_name}"
        logger.info(f"Getting artifact stream: {instance_name} from bucket: {bucket}")

        try:
            data = self.client.get_object(bucket, instance_name)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch artifact stream: {instance_name} from bucket: {bucket}. Error: {e}")
            raise Exception(f"Could not fetch data from bucket: {e}") from e

    def list_artifacts(self, bucket: str) -> List[str]:
        """List all objects in bucket.

        :param bucket: Name of the bucket
        :type bucket: str
        :return: A list of object names
        :rtype: List[str]
        """
        logger.info(f"Listing artifacts in bucket: {bucket}")
        objects = []

        try:
            objs = self.client.list_objects(bucket, prefix=self.project_slug)
            for obj in objs:
                objects.append(obj.object_name)
        except Exception as err:
            logger.error(f"Failed to list artifacts in bucket: {bucket}. Error: {err}")
            raise Exception(f"Could not list models in bucket: {bucket}") from err

        return objects

    def delete_artifact(self, instance_name: str, bucket: str) -> None:
        """Delete object with name instance_name from buckets.

        :param instance_name: The object name
        :type instance_name: str
        :param bucket: Buckets to delete from
        :type bucket: str
        """
        instance_name = f"{self.project_slug}/{instance_name}"
        logger.info(f"Deleting artifact: {instance_name} from bucket: {bucket}")

        try:
            self.client.remove_object(bucket, instance_name)
        except InvalidResponseError as err:
            logger.error(f"Could not delete artifact: {instance_name}. Error: {err}")

    def create_bucket(self, bucket_name: str) -> None:
        """Create a new bucket. If bucket exists, do nothing.

        :param bucket_name: The name of the bucket
        :type bucket_name: str
        """
        logger.info(f"Creating bucket: {bucket_name}")

        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        except InvalidResponseError as err:
            logger.error(f"Failed to create bucket: {bucket_name}. Error: {err}")
            raise
