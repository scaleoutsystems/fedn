"""Module implementing Repository for MinIO."""

import io
from typing import IO, List

from minio import Minio
from minio.error import InvalidResponseError
from urllib3.poolmanager import PoolManager

from fedn.common.log_config import logger
from fedn.network.storage.s3.base import RepositoryBase


class MINIORepository(RepositoryBase):
    """Class implementing Repository for MinIO."""

    client = None

    def __init__(self, config: dict) -> None:
        """Initialize object.

        :param config: Dictionary containing configuration for credentials and bucket names.
        :type config: dict
        """
        super().__init__()
        self.name = "MINIORepository"

        if config["storage_secure_mode"]:
            manager = PoolManager(num_pools=100, cert_reqs="CERT_NONE", assert_hostname=False)
            self.client = Minio(
                "{0}:{1}".format(config["storage_hostname"], config["storage_port"]),
                access_key=config["storage_access_key"],
                secret_key=config["storage_secret_key"],
                secure=config["storage_secure_mode"],
                http_client=manager,
            )
        else:
            self.client = Minio(
                "{0}:{1}".format(config["storage_hostname"], config["storage_port"]),
                access_key=config["storage_access_key"],
                secret_key=config["storage_secret_key"],
                secure=config["storage_secure_mode"],
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
        try:
            return self.client.get_object(bucket, instance_name)
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
        objects = []
        try:
            objs = self.client.list_objects(bucket)
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
