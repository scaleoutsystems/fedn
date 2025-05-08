"""Implementation of the Repository interface for SaaS deployment using boto3."""

import io
import os
from typing import IO, List

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from fedn.common.config import (
    FEDN_OBJECT_STORAGE_ACCESS_KEY,
    FEDN_OBJECT_STORAGE_ENDPOINT,
    FEDN_OBJECT_STORAGE_REGION,
    FEDN_OBJECT_STORAGE_SECRET_KEY,
    FEDN_OBJECT_STORAGE_SECURE_MODE,
    FEDN_OBJECT_STORAGE_VERIFY_SSL,
)
from fedn.common.log_config import logger
from fedn.network.storage.s3.base import RepositoryBase


class SAASRepository(RepositoryBase):
    """Class implementing Repository for SaaS deployment using boto3."""

    def __init__(self, config: dict) -> None:
        """Initialize object.

        :param config: Dictionary containing configuration for credentials and bucket names.
        :type config: dict
        """
        super().__init__()
        self.name = "SAASRepository"
        self.project_slug = os.environ.get("FEDN_JWT_CUSTOM_CLAIM_VALUE", "default_project")

        access_key = config.get("storage_access_key", FEDN_OBJECT_STORAGE_ACCESS_KEY)
        secret_key = config.get("storage_secret_key", FEDN_OBJECT_STORAGE_SECRET_KEY)
        endpoint_url = config.get("storage_endpoint", FEDN_OBJECT_STORAGE_ENDPOINT)
        region_name = config.get("storage_region", FEDN_OBJECT_STORAGE_REGION)
        use_ssl = config.get("storage_secure_mode", FEDN_OBJECT_STORAGE_SECURE_MODE)
        use_ssl = use_ssl.lower() == "true" if isinstance(use_ssl, str) else use_ssl
        verify_ssl = config.get("storage_verify_ssl", FEDN_OBJECT_STORAGE_VERIFY_SSL)
        verify_ssl = verify_ssl.lower() == "true" if isinstance(verify_ssl, str) else verify_ssl

        # Initialize the boto3 client
        common_config = {
            "endpoint_url": endpoint_url,
            "region_name": region_name,
            "use_ssl": use_ssl,
            "verify": verify_ssl,
        }
        logger.info(f"Connection parameters: {common_config}")
        logger.info(f"Keys: {access_key} {secret_key}")

        if access_key and secret_key:
            # Use provided credentials
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version="s3v4"),
                **common_config,
            )
            logger.info(f"Using {self.name} with provided credentials for SaaS storage.")
        else:
            # Use default credentials (e.g., IAM roles, service accounts, or environment variables)
            self.s3_client = boto3.client("s3", config=Config(signature_version="s3v4") ** common_config)
            logger.info(f"Using {self.name} with default credentials for SaaS storage.")

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
                self.s3_client.upload_file(instance, bucket, instance_name)
            else:
                self.s3_client.put_object(Bucket=bucket, Key=instance_name, Body=instance)
            return True
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to upload artifact: {instance_name} to bucket: {bucket}. Error: {e}")
            raise Exception(f"Could not upload artifact: {e}") from e

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
            response = self.s3_client.get_object(Bucket=bucket, Key=instance_name)
            return response["Body"].read()
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to fetch artifact: {instance_name} from bucket: {bucket}. Error: {e}")
            raise Exception(f"Could not fetch artifact: {e}") from e

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
            response = self.s3_client.get_object(Bucket=bucket, Key=instance_name)
            return io.BytesIO(response["Body"].read())
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to fetch artifact stream: {instance_name} from bucket: {bucket}. Error: {e}")
            raise Exception(f"Could not fetch artifact stream: {e}") from e

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
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=self.project_slug)
            for obj in response.get("Contents", []):
                objects.append(obj["Key"])
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to list artifacts in bucket: {bucket}. Error: {e}")
            raise Exception(f"Could not list artifacts: {e}") from e

        return objects

    def delete_artifact(self, instance_name: str, bucket: str) -> None:
        """Delete object with name instance_name from bucket.

        :param instance_name: The object name
        :type instance_name: str
        :param bucket: Bucket to delete from
        :type bucket: str
        """
        instance_name = f"{self.project_slug}/{instance_name}"
        logger.info(f"Deleting artifact: {instance_name} from bucket: {bucket}")

        try:
            self.s3_client.delete_object(Bucket=bucket, Key=instance_name)
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to delete artifact: {instance_name} from bucket: {bucket}. Error: {e}")
            raise Exception(f"Could not delete artifact: {e}") from e

    def create_bucket(self, bucket_name: str) -> None:
        """Create a new bucket. If bucket exists, do nothing.

        :param bucket_name: The name of the bucket
        :type bucket_name: str
        """
        logger.info(f"Creating bucket: {bucket_name}")

        try:
            self.s3_client.create_bucket(Bucket=bucket_name)
        except self.s3_client.exceptions.BucketAlreadyExists:
            logger.info(f"Bucket {bucket_name} already exists.")
        except self.s3_client.exceptions.BucketAlreadyOwnedByYou:
            logger.info(f"Bucket {bucket_name} already owned by you. No action needed.")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to create bucket: {bucket_name}. Error: {e}")
            raise Exception(f"Could not create bucket: {e}") from e
        logger.info(f"Bucket {bucket_name} created successfully.")
