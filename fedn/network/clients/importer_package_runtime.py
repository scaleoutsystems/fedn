"""Contains the PackageRuntime class, used to download, validate, and unpack compute packages."""

import cgi
import os
import tarfile
from typing import Optional, Tuple

import requests

from fedn.common.config import FEDN_AUTH_SCHEME, FEDN_CLIENTCACHE_DIR, FEDN_CONNECT_API_SECURE, FEDN_PACKAGE_EXTRACT_DIR
from fedn.common.log_config import logger
from fedn.network.clients.connect import HTTP_STATUS_NO_CONTENT
from fedn.network.clients.http_status_codes import HTTP_STATUS_OK
from fedn.utils.checksum import sha
from fedn.utils.dispatcher import Dispatcher, _read_yaml_file

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds


def get_compute_package_dir_path() -> str:
    """Get the directory path for the compute package."""
    cachedir = os.path.join(os.getcwd(), FEDN_CLIENTCACHE_DIR)
    packagedir = os.path.join(os.getcwd(), FEDN_PACKAGE_EXTRACT_DIR)

    os.makedirs(cachedir, exist_ok=True)
    os.makedirs(packagedir, exist_ok=True)

    return cachedir, packagedir


class ImporterPackageRuntime:
    """PackageRuntime is used to download, validate, and unpack compute packages.

    :param package_path: Path to compute package.
    :type package_path: str
    """

    def __init__(self, cache_dir, package_path: str) -> None:
        self.pkg_path = package_path
        self.tar_cache_path = os.path.join(cache_dir, "packages")
        os.makedirs(self.tar_cache_path, exist_ok=True)
        self.pkg_name: Optional[str] = None
        self.checksum: Optional[str] = None

        self.startup_path: Optional[str] = None

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
            with requests.get(
                path, stream=True, timeout=REQUEST_TIMEOUT, headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"}, verify=FEDN_CONNECT_API_SECURE
            ) as r:
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
            with requests.get(path, timeout=REQUEST_TIMEOUT, headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"}, verify=FEDN_CONNECT_API_SECURE) as r:
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
                    if "startup.py" in files:
                        logger.info(f"Found startup.py file in {root}")
                        return True, root

                logger.error("No startup.py file found in extracted package!")
                return False, ""
        except Exception as e:
            logger.error(f"Error extracting files: {e}")
            os.remove(os.path.join(self.pkg_path, self.pkg_name))
            return False, ""

    def run_startup(self) -> bool:
        """Get or set the environment."""
        try:
            
        except KeyError:
            logger.info("No startup command found in package. Continuing.")
        except Exception as e:
            logger.error(f"Caught exception: {type(e).__name__}")
            return False

        
        return True

    def init_remote_compute_package(self, url: str, token: str, package_checksum: Optional[str] = None) -> bool:
        """Initialize the remote compute package."""
        result = self.download_compute_package(url, token)
        if not result:
            logger.error("Could not download compute package")
            return False
        result = self.set_checksum(url, token)
        if not result:
            logger.error("Could not set checksum")
            return False

        if package_checksum:
            result = self.validate(package_checksum)
            if not result:
                logger.error("Could not validate compute package")
                return False

        result, path = self.unpack_compute_package()

        if not result:
            logger.error("Could not unpack compute package")
            return False

        logger.info(f"Compute package unpacked to: {path}")

        result = self.set_dispatcher(path)

        if not result:
            logger.error("Could not set dispatcher")
            return False

        logger.info("Dispatcher set")
        result = self.init_dispatcher()

        return True

    def init_local_compute_package(self) -> bool:
        """Initialize the local compute package."""
        path = os.path.join(os.getcwd(), "client")
        result = self.set_dispatcher(path)

        if not result:
            logger.error("Could not set dispatcher")
            return False

        result = self.init_dispatcher()

        logger.info("Dispatcher set")

        return True
