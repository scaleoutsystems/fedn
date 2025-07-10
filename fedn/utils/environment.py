"""Portions of this code are derived from the Apache 2.0 licensed project mlflow (https://mlflow.org/).,
with modifications made by Scaleout Systems AB.
Copyright (c) 2018 Databricks, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import yaml

from fedn.common.log_config import logger
from fedn.utils import PYTHON_VERSION
from fedn.utils.process import _exec_cmd, _join_commands

_REQUIREMENTS_FILE_NAME = "requirements.txt"
_PYTHON_ENV_FILE_NAME = "python_env.yaml"
_PYTHON_ENV_METADATA_FILE_NAME = "env_metadata.txt"
_IS_UNIX = os.name != "nt"


class _PythonEnv:
    def __init__(self, name=None, python=None, build_dependencies=None, dependencies=None):
        """Represents environment information for FEDn compute packages.

        Args:
        ----
            name: Name of environment. If unspecified, defaults to fedn_env
            python: Python version for the environment. If unspecified, defaults to the current
                Python version.
            build_dependencies: List of build dependencies for the environment that must
                be installed before installing ``dependencies``. If unspecified,
                defaults to an empty list.
            dependencies: List of dependencies for the environment. If unspecified, defaults to
                an empty list.

        """
        if name is not None and not isinstance(name, str):
            raise TypeError(f"`name` must be a string but got {type(name)}")
        if python is not None and not isinstance(python, str):
            raise TypeError(f"`python` must be a string but got {type(python)}")
        if build_dependencies is not None and not isinstance(build_dependencies, list):
            raise TypeError(f"`build_dependencies` must be a list but got {type(build_dependencies)}")
        if dependencies is not None and not isinstance(dependencies, list):
            raise TypeError(f"`dependencies` must be a list but got {type(dependencies)}")
        self._name = name or "fedn_env"
        self.python = python or PYTHON_VERSION
        self.build_dependencies = build_dependencies or []
        self.dependencies = dependencies or []
        self._base_path = None

    @property
    def name(self):
        """Name of the environment."""
        return os.path.join(self._name, self.get_sha())

    def set_base_path(self, path):
        self._base_path = path

    @property
    def path(self) -> Path:
        """Get the full path to the environment."""
        if not self._base_path:
            raise ValueError("Base path is not set. Use `set_base_path` to set it.")
        return Path(self._base_path).joinpath(self.name)

    def __str__(self):
        return str(self.to_dict())

    def get_sha(self):
        """Returns a SHA256 hash of the environment configuration."""
        env_str = str(self.to_dict()).encode("utf-8")
        return hashlib.sha256(env_str).hexdigest()

    def remove_fedndependency(self):
        """Remove 'fedn' from dependencies if it exists."""
        self.dependencies = [dep for dep in self.dependencies if dep != "fedn"]
        self.build_dependencies = [dep for dep in self.build_dependencies if dep != "fedn"]

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_yaml(self, path):
        with open(path, "w") as f:
            # Exclude None and empty lists
            data = {k: v for k, v in self.to_dict().items() if v}
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @staticmethod
    def get_dependencies_from_conda_yaml(path):
        raise NotImplementedError

    @classmethod
    def from_conda_yaml(cls, path):
        return cls.from_dict(cls.get_dependencies_from_conda_yaml(path))

    def get_activate_cmd(self):
        """Get the command to activate the environment."""
        paths = ("bin", "activate") if _IS_UNIX else ("Scripts", "activate.bat")
        activate_cmd = self.path.joinpath(*paths)
        activate_cmd = f"source {activate_cmd}" if _IS_UNIX else str(activate_cmd)
        return activate_cmd

    def create_virtualenv(self, env_dir, capture_output=False, use_system_site_packages=False):
        # Created a command to activate the environment
        if not isinstance(env_dir, Path):
            env_dir = Path(env_dir)

        activate_cmd = self.get_activate_cmd(env_dir)

        if env_dir.exists():
            logger.info("Environment %s already exists", env_dir)
            return True

        with remove_on_error(
            env_dir,
            onerror=lambda e: logger.warning(
                "Encountered an unexpected error: %s while creating a virtualenv environment in %s, removing the environment directory...",
                repr(e),
                env_dir,
            ),
        ):
            os.makedirs(env_dir, exist_ok=True)
            logger.info("Creating a new environment in %s with %s", env_dir, sys.executable)
            _exec_cmd(
                [sys.executable, "-m", "virtualenv", "--python", sys.executable] + ["--system-site-packages" if use_system_site_packages else ""] + [env_dir],
                capture_output=capture_output,
            )

            extra_env = {
                # PIP_NO_INPUT=1 makes pip run in non-interactive mode,
                # otherwise pip might prompt "yes or no" and ask stdin input
                "PIP_NO_INPUT": "1",
            }

            logger.info("Installing dependencies")
            for deps in filter(None, [self.build_dependencies, self.dependencies]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_req_file = f"requirements.{uuid.uuid4().hex}.txt"
                    Path(tmpdir).joinpath(tmp_req_file).write_text("\n".join(deps))
                    cmd = _join_commands(activate_cmd, f"python -m pip install -r {tmp_req_file}")
                    _exec_cmd(cmd, capture_output=capture_output, cwd=tmpdir, extra_env=extra_env)

            Path(env_dir).joinpath(_PYTHON_ENV_METADATA_FILE_NAME).write_text(
                f"{self.get_sha()}\n{self.python}\n{'\n'.join(self.build_dependencies or [])}\n{'\n'.join(self.dependencies or [])}"
            )

        return True

    def verify_installed_env(self):
        """Check if the environment metadata file exists and matches the current environment."""
        metadata_file = self.path.joinpath(_PYTHON_ENV_METADATA_FILE_NAME)
        if not metadata_file.exists():
            return False

        with open(metadata_file) as f:
            sha, python_version, *deps = f.read().splitlines()
            if sha != self.get_sha() or python_version != self.python:
                return False

            # Check if dependencies match
            if set(deps) != set(self.build_dependencies + self.dependencies):
                return False

        return True


@contextmanager
def remove_on_error(path: os.PathLike, onerror=None):
    """A context manager that removes a file or directory if an exception is raised during
    execution.
    """
    try:
        yield
    except Exception as e:
        if onerror:
            onerror(e)
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        raise
