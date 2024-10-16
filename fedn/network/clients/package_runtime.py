# This file contains the PackageRuntime class, which is used to download, validate and unpack compute packages.
#
#
import cgi
import os
import tarfile
from typing import Tuple

import requests

from fedn.common.config import FEDN_AUTH_SCHEME
from fedn.common.log_config import logger
from fedn.utils.checksum import sha
from fedn.utils.dispatcher import Dispatcher, _read_yaml_file


class PackageRuntime:
    """PackageRuntime is used to download, validate and unpack compute packages.

    :param package_path: path to compute package
    :type package_path: str
    :param package_dir: directory to unpack compute package
    :type package_dir: str
    """

    def __init__(self, package_path: str):
        self.dispatch_config = {
            "entry_points": {
                "predict": {"command": "python3 predict.py"},
                "train": {"command": "python3 train.py"},
                "validate": {"command": "python3 validate.py"},
            }
        }

        self.pkg_path = package_path
        self.pkg_name = None
        self.checksum = None

    def download_compute_package(self, url: str, token: str, name: str = None) -> bool:
        """Download compute package from controller
        :param host: host of controller
        :param port: port of controller
        :param token: token for authentication
        :param name: name of package
        :return: True if download was successful, None otherwise
        :rtype: bool
        """
        try:
            # TODO: use new endpoint (v1)
            path = f"{url}/download_package?name={name}" if name else f"{url}/download_package"

            with requests.get(path, stream=True, verify=False, headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"}) as r:
                if 200 <= r.status_code < 204:
                    params = cgi.parse_header(r.headers.get("Content-Disposition", ""))[-1]
                    try:
                        self.pkg_name = params["filename"]
                    except KeyError:
                        logger.error("No package returned.")
                        return None
                    r.raise_for_status()
                    with open(os.path.join(self.pkg_path, self.pkg_name), "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

            return True
        except Exception:
            return False

    def set_checksum(self, url: str, token: str, name: str = None) -> bool:
        """Get checksum of compute package from controller
        :param host: host of controller
        :param port: port of controller
        :param token: token for authentication
        :param name: name of package
        :return: checksum of the compute package
        :rtype: str
        """
        try:
            # TODO: use new endpoint (v1)
            path = f"{url}/get_package_checksum?name={name}" if name else f"{url}/get_package_checksum"

            with requests.get(path, verify=False, headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"}) as r:
                if 200 <= r.status_code < 204:
                    data = r.json()
                    try:
                        self.checksum = data["checksum"]
                    except Exception:
                        logger.error("Could not extract checksum.")

            return True
        except Exception:
            return False

    def validate(self, expected_checksum) -> bool:
        """Validate the package against the checksum provided by the controller

        :param expected_checksum: checksum provided by the controller
        :return: True if checksums match, False otherwise
        :rtype: bool
        """
        # crosscheck checksum and unpack if security checks are ok.
        file_checksum = str(sha(os.path.join(self.pkg_path, self.pkg_name)))

        if self.checksum == expected_checksum == file_checksum:
            logger.info("Package validated {}".format(self.checksum))
            return True
        else:
            return False

    def unpack_compute_package(self) -> Tuple[bool, str]:
        """Unpack the compute package

        :return: True if unpacking was successful, False otherwise
        :rtype: bool
        """
        if self.pkg_name:
            f = None
            if self.pkg_name.endswith("tar.gz"):
                f = tarfile.open(os.path.join(self.pkg_path, self.pkg_name), "r:gz")
            if self.pkg_name.endswith(".tgz"):
                f = tarfile.open(os.path.join(self.pkg_path, self.pkg_name), "r:gz")
            if self.pkg_name.endswith("tar.bz2"):
                f = tarfile.open(os.path.join(self.pkg_path, self.pkg_name), "r:bz2")
        else:
            logger.error("Failed to unpack compute package, no pkg_name set." "Has the reducer been configured with a compute package?")
            return False, ""

        try:
            if f:
                f.extractall(self.pkg_path)
                logger.info("Successfully extracted compute package content in {}".format(self.pkg_path))
                # delete the tarball
                logger.info("Deleting temporary package tarball file.")
                f.close()
                os.remove(os.path.join(self.pkg_path, self.pkg_name))
                # search for file fedn.yaml in extracted package
                for root, dirs, files in os.walk(self.pkg_path):
                    if "fedn.yaml" in files:
                        # Get the path to where fedn.yaml is located
                        logger.info("Found fedn.yaml file in {}".format(root))
                        return True, root

                logger.error("No fedn.yaml file found in extracted package!")
                return False, ""
        except Exception:
            logger.error("Error extracting files.")
            # delete the tarball
            os.remove(os.path.join(self.pkg_path, self.pkg_name))
            return False, ""

    def get_dispatcher(self, run_path) -> Dispatcher:
        """Dispatch the compute package

        :param run_path: path to dispatch the compute package
        :type run_path: str
        :return: Dispatcher object
        :rtype: :class:`fedn.utils.dispatcher.Dispatcher`
        """
        try:
            self.dispatch_config = _read_yaml_file(os.path.join(run_path, "fedn.yaml"))
            dispatcher = Dispatcher(self.dispatch_config, run_path)

            return dispatcher
        except Exception:
            return None
