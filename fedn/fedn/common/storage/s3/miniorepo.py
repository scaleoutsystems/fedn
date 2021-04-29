import os
import requests
from .base import Repository
from minio import Minio
from minio.error import InvalidResponseError

import io
import logging
from urllib.parse import urlparse
import uuid
import json

logger = logging.getLogger(__name__)


class MINIORepository(Repository):
    client = None

    def __init__(self, config):
        super().__init__()
        try:
            access_key = config['storage_access_key']
        except Exception:
            access_key = 'minio'
        try:
            secret_key = config['storage_secret_key']
        except Exception:
            secret_key = 'minio123'
        try:
            self.bucket = config['storage_bucket']
        except Exception:
            self.bucket = 'fedn-models'
        try:
            self.context_bucket = config['context_bucket']
        except Exception:
            self.bucket = 'fedn-context'
        try:
            self.secure_mode = bool(config['storage_secure_mode'])
        except Exception:
            self.secure_mode = False

        if not self.secure_mode:
            print("\n\n\nWARNING : S3/MINIO RUNNING IN **INSECURE** MODE! THIS IS NOT FOR PRODUCTION!\n\n\n")

        if self.secure_mode:
            from urllib3.poolmanager import PoolManager
            manager = PoolManager(num_pools=100, cert_reqs='CERT_NONE', assert_hostname=False)
            self.client = Minio("{0}:{1}".format(config['storage_hostname'], config['storage_port']),
                                access_key=access_key,
                                secret_key=secret_key,
                                secure=self.secure_mode, http_client=manager)
        else:
            self.client = Minio("{0}:{1}".format(config['storage_hostname'], config['storage_port']),
                                access_key=access_key,
                                secret_key=secret_key,
                                secure=self.secure_mode)

        # TODO: generalize
        self.context_bucket = 'fedn-context'
        self.create_bucket(self.context_bucket)
        self.create_bucket(self.bucket)

    def create_bucket(self, bucket_name):

        found = self.client.bucket_exists(bucket_name)
        if not found:
            try:
                response = self.client.make_bucket(bucket_name)
            except InvalidResponseError as err:
                raise

    def set_artifact(self, instance_name, instance, is_file=False, bucket=''):
        """ Instance must be a byte-like object. """
        if bucket == '':
            bucket = self.bucket
        if is_file == True:
            self.client.fput_object(bucket, instance_name, instance)
        else:
            try:
                self.client.put_object(bucket, instance_name, io.BytesIO(instance), len(instance))
            except Exception as e:
                raise Exception("Could not load data into bytes {}".format(e))

        return True

    def get_artifact(self, instance_name,bucket=''):

        if bucket == '':
            bucket = self.bucket

        try:
            data = self.client.get_object(bucket, instance_name)
            return data.read()
        except Exception as e:
            raise Exception("Could not fetch data from bucket, {}".format(e))

    def get_artifact_stream(self, instance_name):

        try:
            data = self.client.get_object(self.bucket, instance_name)
            return data
        except Exception as e:
            raise Exception("Could not fetch data from bucket, {}".format(e))

    def list_artifacts(self):
        objects_to_delete = []
        try:
            objs = self.client.list_objects(self.bucket)
            for obj in objs:
                print(obj.object_name)
                objects_to_delete.append(obj.object_name)
        except Exception as e:
            raise Exception("Could not list models in bucket {}".format(self.bucket))
        return objects_to_delete

    def delete_artifact(self, instance_name, bucket=[]):
        if not bucket:
            bucket = self.bucket

        try:
            self.client.remove_object(bucket, instance_name)
        except ResponseError as err:
            print(err)
            print('Could not delete artifact: {}'.format(instance_name))

    def delete_objects(self):
        objects_to_delete = self.list_artifacts()
        try:
            # force evaluation of the remove_objects() call by iterating over
            # the returned value.
            for del_err in self.client.remove_objects(self.bucket, objects_to_delete):
                print("Deletion Error: {}".format(del_err))
        except ResponseError as err:
            print(err)



