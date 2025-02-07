import io
import os

from minio import Minio
from minio.error import InvalidResponseError
from urllib3.poolmanager import PoolManager

from fedn.common.log_config import logger
from fedn.network.storage.s3.base import RepositoryBase


class SAASRepository(RepositoryBase):
    """Class implementing Repository for SaaS deployment."""

    client = None

    def __init__(self, config):
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
        logger.info("keys:")
        logger.info(access_key)
        logger.info(secret_key)
        storage_hostname = os.environ.get("FEDN_STORAGE_HOSTNAME", config["storage_hostname"])
        storage_port = os.environ.get("FEDN_STORAGE_PORT", config["storage_port"])
        storage_secure_mode = os.environ.get("FEDN_STORAGE_SECURE_MODE", config["storage_secure_mode"])
        logger.info(f"storage secure mode: {storage_secure_mode}")
        #storage_secure_mode = storage_secure_mode.lower() == "true"

        # if storage_secure_mode:
        manager = PoolManager(num_pools=100, cert_reqs="CERT_NONE", assert_hostname=False)
        logger.info("connection to host: ")
        logger.info(f"{storage_hostname}:{storage_port}")
        self.client = Minio(
            f"{storage_hostname}:{storage_port}",
            access_key=access_key,
            secret_key=secret_key,
            secure=storage_secure_mode,
            http_client=manager,
        )
        # else:
        #     self.client = Minio(
        #         f"{storage_hostname}:{storage_port}",
        #         access_key=access_key,
        #         secret_key=secret_key,
        #         secure=storage_secure_mode,
        #     )

    def set_artifact(self, instance_name, instance, bucket, is_file=False):
        instance_name = f"{self.project_slug}/{instance_name}"
        logger.info(instance_name)
        if is_file:
            logger.info("writing file to bucket")
            logger.info(bucket)
            try:
                result = self.client.fput_object(bucket, instance_name, instance)
                logger.info(result)
            except Exception as e:
                logger.info("Failed to upload file.")
                logger.info(e)
        else:
            try:
                self.client.put_object(bucket, instance_name, io.BytesIO(instance), len(instance))
            except Exception as e:
                raise Exception("Could not load data into bytes {}".format(e))

        return True

    def get_artifact(self, instance_name, bucket):
        instance_name = f"{self.project_slug}/{instance_name}"
        try:
            data = self.client.get_object(bucket, instance_name)
            return data.read()
        except Exception as e:
            raise Exception("Could not fetch data from bucket, {}".format(e))
        finally:
            data.close()
            data.release_conn()

    def get_artifact_stream(self, instance_name, bucket):
        instance_name = f"{self.project_slug}/{instance_name}"
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
            objs = self.client.list_objects(bucket, prefix=self.project_slug)
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
            instance_name = f"{self.project_slug}/{instance_name}"
            self.client.remove_object(bucket, instance_name)
        except InvalidResponseError as err:
            logger.error("Could not delete artifact: {0} err: {1}".format(instance_name, err))
            pass

    def create_bucket(self, bucket_name):
        pass
