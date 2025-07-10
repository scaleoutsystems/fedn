"""Contains the PackageRuntime class, used to download, validate, and unpack compute packages."""

import os
import sys
from pathlib import Path
from typing import Optional

from fedn.common.config import FEDN_ARCHIVE_DIR, FEDN_PACKAGE_EXTRACT_DIR
from fedn.common.log_config import logger
from fedn.network.clients.package_runtime import PackageRuntime
from fedn.utils.environment import _PythonEnv

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds


def get_compute_package_dir_path() -> str:
    """Get the directory path for the compute package."""
    full_package_path = os.path.join(os.getcwd(), FEDN_PACKAGE_EXTRACT_DIR)
    full_archive_path = os.path.join(os.getcwd(), FEDN_ARCHIVE_DIR)

    os.makedirs(full_package_path, exist_ok=True)
    os.makedirs(full_archive_path, exist_ok=True)

    return full_package_path, full_archive_path


class ImporterPackageRuntime(PackageRuntime):
    """ImporterPackageRuntime is used to download, validate, and unpack compute packages.

    :param package_path: Path to compute package.
    :type package_path: str
    """

    def __init__(self, package_path: str, archive_path: str) -> None:
        """Initialize the PackageRuntime."""
        super().__init__(package_path, archive_path)
        self.python_env: Optional[_PythonEnv] = None

    def init_env_runtime(self):
        if self.config is None:
            logger.error("Package runtime is not initialized.")
            return False
        try:
            python_env_yaml_path = self.config.get("python_env")
            if python_env_yaml_path:
                logger.info(f"Initializing Python environment from configuration: {python_env_yaml_path}")
                python_env_yaml_path = Path(self._target_path).joinpath(python_env_yaml_path)
                self.python_env = _PythonEnv.from_yaml(python_env_yaml_path)
                self.python_env.remove_fedndependency()
                self.python_env.set_base_path(self._target_path)
                if not self.python_env.path.exists():
                    self.python_env.create_virtualenv(self.python_env.path, capture_output=True, use_system_site_packages=True)
                if not self.python_env.verify_installed_env():
                    logger.error(f"Python environment at {self.python_env.path} is not valid.")
                    raise RuntimeError(f"Invalid Python environment at {self.python_env.path}.")
            else:
                logger.info("No Python environment specified in the configuration, using the system Python.")
                self.python_env = None
        except Exception as e:
            logger.error(f"Error initializing Python environment from configuration: {e}")
            self.python_env = None

    def run_startup(self, fedn_client):
        """Run the client startup script."""
        if self.config is None:
            logger.error("Package runtime is not initialized.")
            return False

        original_sys_path = sys.path.copy()
        try:
            # Add the package path to sys.path
            sys.path.insert(0, self._target_path)
            entrypoint = self.config.get("entry_points")
            if entrypoint:
                startup_py = entrypoint.get("startup")
            if not startup_py:
                logger.info("No startup entrypoint defined in the configuration, using default 'startup.py'.")
                startup_py = "startup.py"

            if not Path(self._target_path).joinpath(startup_py).exists():
                logger.error(f"Startup script {startup_py} not found in the package directory.")
                raise FileNotFoundError(f"Startup script {startup_py} not found.")

            startup_module = Path(self._target_path).joinpath(startup_py).stem
            startup = __import__(startup_module)

            startup.startup(fedn_client)
        except Exception as e:
            logger.error(f"Error during client startup: {e}")
            return False
        finally:
            # Restore the original sys.path
            sys.path = original_sys_path

        return True

    def run_build(self):
        """Run the build script."""
        if self.config is None:
            logger.error("Package runtime is not initialized.")
            return False

        original_sys_path = sys.path.copy()
        try:
            sys.path.insert(0, self._target_path)
            entrypoint = self.config.get("entry_points")
            if entrypoint:
                build_py = entrypoint.get("build")
            if not build_py:
                logger.info("No build entrypoint defined in the configuration, using default 'build.py'.")
                build_py = "build.py"

            if not Path(self._target_path).joinpath(build_py).exists():
                logger.error(f"Build script {build_py} not found in the package directory.")
                raise FileNotFoundError(f"Build script {build_py} not found.")

            build_module = Path(self._target_path).joinpath(build_py).stem
            build = __import__(build_module)

            build.build()
        except Exception as e:
            logger.error(f"Error during build: {e}")
            return False
        finally:
            # Restore the original sys.path
            sys.path = original_sys_path

        return True
