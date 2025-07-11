import cgi
import os
import tarfile
from typing import Optional

import requests

from fedn.common.config import FEDN_ARCHIVE_DIR, FEDN_AUTH_SCHEME, FEDN_CONNECT_API_SECURE, FEDN_PACKAGE_EXTRACT_DIR
from fedn.common.log_config import logger
from fedn.network.clients.http_status_codes import HTTP_STATUS_NO_CONTENT, HTTP_STATUS_OK
from fedn.utils.checksum import sha
from fedn.utils.yaml import read_yaml_file

REQUEST_TIMEOUT = 10  # seconds  # Default timeout for requests


def get_compute_package_dir_path() -> str:
    """Get the directory path for the compute package."""
    full_package_path = os.path.join(os.getcwd(), FEDN_PACKAGE_EXTRACT_DIR)
    full_archive_path = os.path.join(os.getcwd(), FEDN_ARCHIVE_DIR)

    os.makedirs(full_package_path, exist_ok=True)
    os.makedirs(full_archive_path, exist_ok=True)

    return full_package_path, full_archive_path


class PackageRuntime:
    def __init__(self, package_path: str, archive_path: str) -> None:
        """Initialize the PackageRuntime."""
        self.pkg_path = package_path
        self.tar_path = os.path.join(archive_path, "packages")
        os.makedirs(self.tar_path, exist_ok=True)

        self.pkg_name: Optional[str] = None
        self._checksum: Optional[str] = None

        self._target_path: Optional[str] = None
        self._target_name = "fedn.yaml"

        self.config = None

    def _download_compute_package(self, url: str, token: str, name: Optional[str] = None) -> bool:
        """Download compute package from controller.

        :param url: URL of the controller.
        :param token: Token for authentication.
        :param name: Name of the package.
        :return: True if download was successful, False otherwise.
        :rtype: bool
        """
        try:
            url = f"{url}/api/v1/packages/download?name={name}" if name else f"{url}/api/v1/packages/download"
            with requests.get(
                url, stream=True, timeout=REQUEST_TIMEOUT, headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"}, verify=FEDN_CONNECT_API_SECURE
            ) as r:
                if HTTP_STATUS_OK <= r.status_code < HTTP_STATUS_NO_CONTENT:
                    params = cgi.parse_header(r.headers.get("Content-Disposition", ""))[-1]
                    try:
                        self.pkg_name = params["filename"]
                        r.raise_for_status()
                        with open(os.path.join(self.tar_path, self.pkg_name), "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        return True

                    except KeyError:
                        logger.error("No package returned.")
                        return False
                else:
                    logger.error(f"Failed to download package: {r.status_code} {r.reason}")
                    return False
        except Exception as e:
            logger.error(f"Unknown error downloading package: {e}")
            return False

    def _fetch_package_checksum(self, url: str, token: str) -> bool:
        """Get checksum of compute package from controller.

        :param url: URL of the controller.
        :param token: Token for authentication.
        :param name: Name of the package.
        :return: True if checksum was set successfully, False otherwise.
        :rtype: bool
        """
        try:
            path = f"{url}/api/v1/packages/checksum?name={self.pkg_name}"
            with requests.get(path, timeout=REQUEST_TIMEOUT, headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"}, verify=FEDN_CONNECT_API_SECURE) as r:
                if HTTP_STATUS_OK <= r.status_code < HTTP_STATUS_NO_CONTENT:
                    data = r.json()
                    try:
                        self._checksum = data["checksum"]
                    except KeyError:
                        logger.error("Could not extract checksum.")
            return True
        except Exception:
            return False

    def validate_compute_package(self, url: str, token: str) -> bool:
        """Validate the package against the checksum provided by the controller.

        :param expected_checksum: Checksum provided by the controller.
        :return: True if checksums match, False otherwise.
        :rtype: bool
        """
        try:
            file_checksum = str(sha(os.path.join(self.tar_path, self.pkg_name)))
        except FileNotFoundError:
            logger.error(f"Package file {self.pkg_name} not found in {self.tar_path}.")
            return False

        success = self._fetch_package_checksum(url, token)
        if not success:
            logger.error("Failed to fetch package checksum from controller.")
            return False

        if self._checksum == file_checksum:
            logger.info(f"Package validated {self._checksum}")
            return True
        return False

    def _unpack_compute_package(self) -> Optional[str]:
        """Unpack the compute package.

        :return: Tuple containing a boolean indicating success and the path to the unpacked package.
        :rtype: Tuple[bool, str]
        """
        if not self.pkg_name:
            logger.error("Failed to unpack compute package, no pkg_name set. Has the reducer been configured with a compute package?")
            return False, ""

        try:
            if self.pkg_name.endswith(("tar.gz", ".tgz", "tar.bz2")):
                pkg_path = os.path.join(self.pkg_path, self.pkg_name)
                with tarfile.open(pkg_path, "r:*") as f:
                    for member in f.getmembers():
                        f.extract(member, self.pkg_path)
                logger.info(f"Successfully extracted compute package content in {self.pkg_path}")
                return self.find_target_path(self.pkg_path)
            else:
                return None
        except Exception as e:
            logger.error(f"Error extracting files: {e}")
            os.remove(os.path.join(self.pkg_path, self.pkg_name))
            return None

    def find_target_path(self, path) -> Optional[str]:
        for root, _, files in os.walk(os.path.join(path, "")):
            if self._target_name in files:
                logger.info(f"Found {self._target_name} file in {root}")
                return root
        logger.error(f"No {self._target_name} file found in {path}!")
        return None

    def load_local_compute_package(self, pkg_path) -> bool:
        """Initialize the local compute package."""
        path = self.find_target_path(pkg_path)
        if not path:
            logger.error(f"Could not find {self._target_name} in the provided package path.")
            return False

        logger.info(f"Using compute package at: {path}")
        self._target_path = path
        if not self._load_fednyaml():
            logger.error("Failed to load fedn.yaml configuration file.")
            self._target_path = None
            return False
        return True

    def load_remote_compute_package(self, url: str, token: str, pkg_name: Optional[str] = None, validate: bool = True) -> bool:
        """Initialize the remote compute package."""
        do_download = True
        if pkg_name and os.path.exists(os.path.join(self.tar_path, pkg_name)):
            # Package already exists
            logger.info(f"Compute package {pkg_name} already exists in {self.tar_path}.")
            self.pkg_name = pkg_name
            if validate:
                result = self.validate_compute_package(url, token, self.pkg_name)
                if not result:
                    logger.warning("Already downloaded compute package failed validation.")
                else:
                    logger.info("Already downloaded compute package passed validation.")
                    do_download = False
            else:
                logger.info("Skipping validation of already downloaded compute package.")
                do_download = False

        if do_download:
            result = self._download_compute_package(url, token, pkg_name)
            if not result:
                logger.error("Could not download compute package")
                return False

            if validate:
                result = self.validate_compute_package(url, token)
                if not result:
                    logger.error("Could not validate compute package")
                    return False

        path = self._unpack_compute_package()

        if not path:
            logger.error("Could not unpack compute package")
            return False

        logger.info(f"Compute package unpacked to: {path}")
        self._target_path = path
        if not self._load_fednyaml():
            logger.error("Failed to load fedn.yaml configuration file.")
            self._target_path = None
            return False
        return True

    def _load_fednyaml(self):
        """Load the target configuration file."""
        logger.info(f"Reading {self._target_name} configuration file.")
        self.config = read_yaml_file(os.path.join(self._target_path, self._target_name))
        if not self.config:
            logger.error(f"Configuration file {os.path.join(self._target_path, self._target_name)} not found or is empty.")
            return False
        return True

    def run_startup(self, *args, **kwargs):
        raise NotImplementedError("The start method should be implemented in subclasses.")
