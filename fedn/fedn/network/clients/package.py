# This file contains the PackageRuntime class, which is used to download, validate and unpack compute packages.
#
#
import cgi
import os
import tarfile
from distutils.dir_util import copy_tree

import requests
import yaml

from fedn.common.log_config import logger
from fedn.utils.checksum import sha
from fedn.utils.dispatcher import Dispatcher


class PackageRuntime:
    """ PackageRuntime is used to download, validate and unpack compute packages.

    :param package_path: path to compute package
    :type package_path: str
    :param package_dir: directory to unpack compute package
    :type package_dir: str
    """

    def __init__(self, package_path, package_dir):

        self.dispatch_config = {'entry_points':
                                {'predict': {'command': 'python3 predict.py'},
                                 'train': {'command': 'python3 train.py'},
                                 'validate': {'command': 'python3 validate.py'}}}

        self.pkg_path = package_path
        self.pkg_name = None
        self.dir = package_dir
        self.checksum = None
        self.expected_checksum = None

    def download(self, host, port, token, force_ssl=False, secure=False, name=None):
        """ Download compute package from controller

        :param host: host of controller
        :param port: port of controller
        :param token: token for authentication
        :param name: name of package
        :return: True if download was successful, None otherwise
        :rtype: bool
        """
        # for https we assume a an ingress handles permanent redirect (308)
        if force_ssl:
            scheme = "https"
        else:
            scheme = "http"
        if port:
            path = f"{scheme}://{host}:{port}/download_package"
        else:
            path = f"{scheme}://{host}/download_package"
        if name:
            path = path + "?name={}".format(name)

        with requests.get(path, stream=True, verify=False, headers={'Authorization': 'Token {}'.format(token)}) as r:
            if 200 <= r.status_code < 204:

                params = cgi.parse_header(
                    r.headers.get('Content-Disposition', ''))[-1]
                try:
                    self.pkg_name = params['filename']
                except KeyError:
                    logger.error("No package returned.")
                    return None
                r.raise_for_status()
                with open(os.path.join(self.pkg_path, self.pkg_name), 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        if port:
            path = f"{scheme}://{host}:{port}/get_package_checksum"
        else:
            path = f"{scheme}://{host}/get_package_checksum"

        if name:
            path = path + "?name={}".format(name)
        with requests.get(path, verify=False, headers={'Authorization': 'Token {}'.format(token)}) as r:
            if 200 <= r.status_code < 204:

                data = r.json()
                try:
                    self.checksum = data['checksum']
                except Exception:
                    logger.error("Could not extract checksum.")

        return True

    def validate(self, expected_checksum):
        """ Validate the package against the checksum provided by the controller

        :param expected_checksum: checksum provided by the controller
        :return: True if checksums match, False otherwise
        :rtype: bool
        """
        self.expected_checksum = expected_checksum

        # crosscheck checksum and unpack if security checks are ok.
        file_checksum = str(sha(os.path.join(self.pkg_path, self.pkg_name)))

        if self.checksum == self.expected_checksum == file_checksum:
            logger.info("Package validated {}".format(self.checksum))
            return True
        else:
            return False

    def unpack(self):
        """ Unpack the compute package

        :return: True if unpacking was successful, False otherwise
        :rtype: bool
        """
        if self.pkg_name:
            f = None
            if self.pkg_name.endswith('tar.gz'):
                f = tarfile.open(os.path.join(
                    self.pkg_path, self.pkg_name), 'r:gz')
            if self.pkg_name.endswith('.tgz'):
                f = tarfile.open(os.path.join(
                    self.pkg_path, self.pkg_name), 'r:gz')
            if self.pkg_name.endswith('tar.bz2'):
                f = tarfile.open(os.path.join(
                    self.pkg_path, self.pkg_name), 'r:bz2')
        else:
            logger.error(
                "Failed to unpack compute package, no pkg_name set."
                "Has the reducer been configured with a compute package?"
            )
            return False

        os.getcwd()
        try:
            os.chdir(self.dir)

            if f:
                f.extractall()
                logger.info("Successfully extracted compute package content in {}".format(self.dir))
                return True
        except Exception:
            logger.error("Error extracting files.")
            return False

    def dispatcher(self, run_path):
        """ Dispatch the compute package

        :param run_path: path to dispatch the compute package
        :type run_path: str
        :return: Dispatcher object
        :rtype: :class:`fedn.utils.dispatcher.Dispatcher`
        """
        from_path = os.path.join(os.getcwd(), 'client')

        # preserve_times=False ensures compatibility with Gramine LibOS
        copy_tree(from_path, run_path, preserve_times=False)

        try:
            cfg = None
            with open(os.path.join(run_path, 'fedn.yaml'), 'rb') as config_file:

                cfg = yaml.safe_load(config_file.read())
                self.dispatch_config = cfg

        except Exception:
            logger.error(
                "Error trying to load and unpack dispatcher config - trying default")

        dispatcher = Dispatcher(self.dispatch_config, run_path)

        return dispatcher
