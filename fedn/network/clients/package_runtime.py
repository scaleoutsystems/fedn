"""Contains the PackageRuntime class, used to download, validate, and unpack compute packages."""

import cgi
import os
import tarfile
from typing import Optional, Tuple

import requests

from fedn.common.config import FEDN_AUTH_SCHEME, FEDN_CONNECT_API_SECURE
from fedn.common.log_config import logger
from fedn.utils.checksum import sha
from fedn.utils.dispatcher import Dispatcher, _read_yaml_file

# Constants for HTTP status codes
HTTP_STATUS_OK = 200
HTTP_STATUS_NO_CONTENT = 204

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds


class PackageRuntime:
    """PackageRuntime is used to download, validate, and unpack compute packages.

    :param package_path: Path to compute package.
    :type package_path: str
    """

    def __init__(self, package_path: str) -> None:
        """Initialize the PackageRuntime."""
        self.dispatch_config = {
            "entry_points": {
                "predict": {"command": "python3 predict.py"},
                "train": {"command": "python3 train.py"},
                "validate": {"command": "python3 validate.py"},
            }
        }

        self.pkg_path = package_path
        self.pkg_name: Optional[str] = None
        self.checksum: Optional[str] = None

    def download_compute_package(self, url: str, token: str, name: Optional[str] = None) -> bool:
        """Download compute package from controller.

        :param url: URL of the controller.
        :param token: Token for authentication.
        :param name: Name of the package.
        :return: True if download was successful, False otherwise.
        :rtype: bool
        """
        try:
            path = f"{url}/api/v1/packages/download?name={name}" if name else f"{url}/api/v1/packages/download"
            with requests.get(path,
                              stream=True,
                              timeout=REQUEST_TIMEOUT,
                              headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"},
                              verify=FEDN_CONNECT_API_SECURE) as r:
                if HTTP_STATUS_OK <= r.status_code < HTTP_STATUS_NO_CONTENT:
                    params = cgi.parse_header(r.headers.get("Content-Disposition", ""))[-1]
                    try:
                        self.pkg_name = params["filename"]
                    except KeyError:
                        logger.error("No package returned.")
                        return False
                    r.raise_for_status()
                    with open(os.path.join(self.pkg_path, self.pkg_name), "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

            return True
        except Exception:
            return False

    def set_checksum(self, url: str, token: str, name: Optional[str] = None) -> bool:
        """Get checksum of compute package from controller.

        :param url: URL of the controller.
        :param token: Token for authentication.
        :param name: Name of the package.
        :return: True if checksum was set successfully, False otherwise.
        :rtype: bool
        """
        try:
            path = f"{url}/api/v1/packages/checksum?name={name}" if name else f"{url}/api/v1/packages/checksum"
            with requests.get(path,
                              timeout=REQUEST_TIMEOUT,
                              headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"},
                              verify=FEDN_CONNECT_API_SECURE) as r:
                if HTTP_STATUS_OK <= r.status_code < HTTP_STATUS_NO_CONTENT:
                    data = r.json()
                    try:
                        self.checksum = data["checksum"]
                    except KeyError:
                        logger.error("Could not extract checksum.")

            return True
        except Exception:
            return False

    def validate(self, expected_checksum: str) -> bool:
        """Validate the package against the checksum provided by the controller.

        :param expected_checksum: Checksum provided by the controller.
        :return: True if checksums match, False otherwise.
        :rtype: bool
        """
        file_checksum = str(sha(os.path.join(self.pkg_path, self.pkg_name)))

        if self.checksum == expected_checksum == file_checksum:
            logger.info(f"Package validated {self.checksum}")
            return True
        return False

    def unpack_compute_package(self) -> Tuple[bool, str]:
        """Unpack the compute package.

        :return: Tuple containing a boolean indicating success and the path to the unpacked package.
        :rtype: Tuple[bool, str]
        """
        if not self.pkg_name:
            logger.error("Failed to unpack compute package, no pkg_name set. Has the reducer been configured with a compute package?")
            return False, ""

        try:
            if self.pkg_name.endswith(("tar.gz", ".tgz", "tar.bz2")):
                with tarfile.open(os.path.join(self.pkg_path, self.pkg_name), "r:*") as f:
                    for member in f.getmembers():
                        f.extract(member, self.pkg_path)
                logger.info(f"Successfully extracted compute package content in {self.pkg_path}")
                logger.info("Deleting temporary package tarball file.")
                os.remove(os.path.join(self.pkg_path, self.pkg_name))

                for root, _, files in os.walk(os.path.join(self.pkg_path, "")):
                    if "fedn.yaml" in files:
                        logger.info(f"Found fedn.yaml file in {root}")
                        return True, root

                logger.error("No fedn.yaml file found in extracted package!")
                return False, ""
        except Exception as e:
            logger.error(f"Error extracting files: {e}")
            os.remove(os.path.join(self.pkg_path, self.pkg_name))
            return False, ""

    def get_dispatcher(self, run_path: str) -> Optional[Dispatcher]:
        """Dispatch the compute package.

        :param run_path: Path to dispatch the compute package.
        :type run_path: str
        :return: Dispatcher object or None if an error occurred.
        :rtype: Optional[Dispatcher]
        """
        try:
            self.dispatch_config = _read_yaml_file(os.path.join(run_path, "fedn.yaml"))
            return Dispatcher(self.dispatch_config, run_path)
        except Exception as e:
            logger.error(f"Error getting dispatcher: {e}")
            return None
