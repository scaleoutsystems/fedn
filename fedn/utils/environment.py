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

import yaml

from fedn.utils import PYTHON_VERSION

_REQUIREMENTS_FILE_NAME = "requirements.txt"
_PYTHON_ENV_FILE_NAME = "python_env.yaml"


class _PythonEnv:
    BUILD_PACKAGES = ("pip", "setuptools", "wheel")

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
        self.name = name or "fedn_env"
        self.python = python or PYTHON_VERSION
        self.build_dependencies = build_dependencies or []
        self.dependencies = dependencies or []

    def __str__(self):
        return str(self.to_dict())

    @classmethod
    def current(cls):
        return cls(
            python=PYTHON_VERSION,
            build_dependencies=cls.get_current_build_dependencies(),
            dependencies=[f"-r {_REQUIREMENTS_FILE_NAME}"],
        )

    @staticmethod
    def _get_package_version(package_name):
        try:
            return __import__(package_name).__version__
        except (ImportError, AttributeError, AssertionError):
            return None

    @staticmethod
    def get_current_build_dependencies():
        build_dependencies = []
        for package in _PythonEnv.BUILD_PACKAGES:
            version = _PythonEnv._get_package_version(package)
            dep = (package + "==" + version) if version else package
            build_dependencies.append(dep)
        return build_dependencies

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
