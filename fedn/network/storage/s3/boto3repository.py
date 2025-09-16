"""Module implementing Repository for Amazon S3 using boto3."""

import io
from typing import IO, List

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from fedn.common.log_config import logger
from fedn.network.storage.s3.base import RepositoryBase


class Boto3Repository(RepositoryBase):
    """Class implementing Repository for Amazon S3 using boto3."""

    def __init__(self, config: dict) -> None:
        """Initialize object."""
        super().__init__()
        self.name = "Boto3Repository"

        common_config = {
            "use_ssl": config.get("storage_secure_mode", True),
            "verify": config.get("storage_verify_ssl", True),
        }

        access_key = config.get("storage_access_key")
        secret_key = config.get("storage_secret_key")

        if access_key and secret_key:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=config.get("storage_region", "eu-west-1"),
                endpoint_url=config.get("storage_endpoint", "http://minio:9000"),
                **common_config,
            )
        else:
            # Use default credentials (IAM role via service account, environment variables, etc.)

            self.s3_client = boto3.client("s3", **common_config)

        logger.info(f"Using {self.name} for S3 storage.")

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
                logger.info(f"Uploading file to bucket: {bucket} with key: {instance_name}")
                self.s3_client.upload_file(Filename=instance, Bucket=bucket, Key=instance_name)
            else:
                logger.info(f"Uploading object to bucket: {bucket} with key: {instance_name}")
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
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=instance_name)
            return response["Body"]
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
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to list artifacts in bucket: {bucket}. Error: {e}")
            raise Exception(f"Could not list artifacts: {e}") from e

    def delete_artifact(self, instance_name: str, bucket: str) -> None:
        """Delete object with name instance_name from bucket.

        :param instance_name: The object name
        :type instance_name: str
        :param bucket: Bucket to delete from
        :type bucket: str
        """
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
        try:
            self.s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} created successfully.")
        except self.s3_client.exceptions.BucketAlreadyExists:
            logger.info(f"Bucket {bucket_name} already exists. No new bucket was created.")
        except self.s3_client.exceptions.BucketAlreadyOwnedByYou:
            logger.info(f"Bucket {bucket_name} already owned by you. No new bucket was created.")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to create bucket: {bucket_name}. Error: {e}")
            raise Exception(f"Could not create bucket: {e}") from e
