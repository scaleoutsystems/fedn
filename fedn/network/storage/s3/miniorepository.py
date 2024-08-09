import io

from minio import Minio
from minio.error import InvalidResponseError
from urllib3.poolmanager import PoolManager

from fedn.common.log_config import logger
from fedn.network.storage.s3.base import RepositoryBase


class MINIORepository(RepositoryBase):
    """Class implementing Repository for MinIO."""

    client = None

    def __init__(self, config):
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

    def set_artifact(self, instance_name, instance, bucket, is_file=False):
        if is_file:
            self.client.fput_object(bucket, instance_name, instance)
        else:
            try:
                self.client.put_object(bucket, instance_name, io.BytesIO(instance), len(instance))
            except Exception as e:
                raise Exception("Could not load data into bytes {}".format(e))

        return True

    def get_artifact(self, instance_name, bucket):
        try:
            data = self.client.get_object(bucket, instance_name)
            return data.read()
        except Exception as e:
            raise Exception("Could not fetch data from bucket, {}".format(e))
        finally:
            data.close()
            data.release_conn()

    def get_artifact_stream(self, instance_name, bucket):
        try:
            data = self.client.get_object(bucket, instance_name)
            return data
        except Exception as e:
            raise Exception("Could not fetch data from bucket, {}".format(e))

    def list_artifacts(self, bucket):
        """List all objects in bucket.

        :param bucket: Name of the bucket
        :type bucket: str
        :return: A list of object names
        """
        objects = []
        try:
            objs = self.client.list_objects(bucket)
            for obj in objs:
                objects.append(obj.object_name)
        except Exception:
            raise Exception("Could not list models in bucket {}".format(bucket))
        return objects

    def delete_artifact(self, instance_name, bucket):
        """Delete object with name instance_name from buckets.

        :param instance_name: The object name
        :param bucket: Buckets to delete from
        :type bucket: str
        """
        try:
            self.client.remove_object(bucket, instance_name)
        except InvalidResponseError as err:
            logger.error("Could not delete artifact: {0} err: {1}".format(instance_name, err))
            pass

    def create_bucket(self, bucket_name):
        """Create a new bucket. If bucket exists, do nothing.

        :param bucket_name: The name of the bucket
        :type bucket_name: str
        """
        found = self.client.bucket_exists(bucket_name)

        if not found:
            try:
                self.client.make_bucket(bucket_name)
            except InvalidResponseError:
                raise
