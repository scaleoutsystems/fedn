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
from fedn.utils.environment import _PythonEnv
from fedn.utils.process import _exec_cmd, _join_commands

_IS_UNIX = os.name != "nt"


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


def _install_python(version, pyenv_root=None, capture_output=False):
    """Installs a specified version of python with pyenv and returns a path to the installed python
    binary.
    """
    raise NotImplementedError("This function is not implemented yet.")


def _is_virtualenv_available():
    """Returns True if virtualenv is available, otherwise False.
    """
    return shutil.which("virtualenv") is not None


def _validate_virtualenv_is_available():
    """Validates virtualenv is available. If not, throws an `Exception` with a brief instruction
    on how to install virtualenv.
    """
    if not _is_virtualenv_available():
        raise Exception("Could not find the virtualenv binary. Run `pip install virtualenv` to install " "virtualenv.")


def _get_virtualenv_extra_env_vars(env_root_dir=None):
    extra_env = {
        # PIP_NO_INPUT=1 makes pip run in non-interactive mode,
        # otherwise pip might prompt "yes or no" and ask stdin input
        "PIP_NO_INPUT": "1",
    }
    return extra_env


def _get_python_env(python_env_file):
    """Parses a python environment file and returns a dictionary with the parsed content.
    """
    if os.path.exists(python_env_file):
        return _PythonEnv.from_yaml(python_env_file)


def _create_virtualenv(python_bin_path, env_dir, python_env, extra_env=None, capture_output=False):
    # Created a command to activate the environment
    paths = ("bin", "activate") if _IS_UNIX else ("Scripts", "activate.bat")
    activate_cmd = env_dir.joinpath(*paths)
    activate_cmd = f"source {activate_cmd}" if _IS_UNIX else str(activate_cmd)

    if env_dir.exists():
        logger.info("Environment %s already exists", env_dir)
        return activate_cmd

    with remove_on_error(
        env_dir,
        onerror=lambda e: logger.warning(
            "Encountered an unexpected error: %s while creating a virtualenv environment in %s, " "removing the environment directory...",
            repr(e),
            env_dir,
        ),
    ):
        logger.info("Creating a new environment in %s with %s", env_dir, python_bin_path)
        _exec_cmd(
            [sys.executable, "-m", "virtualenv", "--python", python_bin_path, env_dir],
            capture_output=capture_output,
        )

        logger.info("Installing dependencies")
        for deps in filter(None, [python_env.build_dependencies, python_env.dependencies]):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_req_file = f"requirements.{uuid.uuid4().hex}.txt"
                Path(tmpdir).joinpath(tmp_req_file).write_text("\n".join(deps))
                cmd = _join_commands(activate_cmd, f"python -m pip install -r {tmp_req_file}")
                _exec_cmd(cmd, capture_output=capture_output, cwd=tmpdir, extra_env=extra_env)

    return activate_cmd


def _read_yaml_file(file_path):
    try:
        cfg = None
        with open(file_path, "rb") as config_file:
            cfg = yaml.safe_load(config_file.read())

    except Exception as e:
        logger.error(f"Error trying to read yaml file: {file_path}")
        raise e
    return cfg


class Dispatcher:
    """Dispatcher class for compute packages.

    :param config: The configuration.
    :type config: dict
    :param dir: The directory to dispatch to.
    :type dir: str
    """

    def __init__(self, config, project_dir):
        """Initialize the dispatcher."""
        self.config = config
        self.project_dir = project_dir
        self.activate_cmd = ""
        self.python_env_path = ""

    def _get_or_create_python_env(self, capture_output=False, pip_requirements_override=None):
        python_env = self.config.get("python_env", "")
        if not python_env:
            logger.info("No python_env specified in the configuration, using the system Python.")
            return python_env
        else:
            python_env_path = os.path.join(self.project_dir, python_env)
            if not os.path.exists(python_env_path):
                raise Exception("Compute package specified python_env file %s, but no such " "file was found." % python_env_path)
            python_env = _get_python_env(python_env_path)

        extra_env = _get_virtualenv_extra_env_vars()
        env_dir = Path(self.project_dir) / Path(python_env.name)
        self.python_env_path = env_dir
        try:
            python_bin_path = _install_python(python_env.python, capture_output=True)
        except NotImplementedError:
            logger.warning("Failed to install Python: %s", python_env.python)
            logger.warning("Python version installation is not implemented yet.")
            logger.info(f"Using the system Python version: {PYTHON_VERSION}")
            python_bin_path = Path(sys.executable)

        try:
            activate_cmd = _create_virtualenv(
                python_bin_path,
                env_dir,
                python_env,
                extra_env=extra_env,
                capture_output=capture_output,
            )
            # Install additional dependencies specified by `requirements_override`
            if pip_requirements_override:
                logger.info("Installing additional dependencies specified by " f"pip_requirements_override: {pip_requirements_override}")
                cmd = _join_commands(
                    activate_cmd,
                    f"python -m pip install --quiet -U {' '.join(pip_requirements_override)}",
                )
                _exec_cmd(cmd, capture_output=capture_output, extra_env=extra_env)
            self.activate_cmd = activate_cmd
            return activate_cmd
        except Exception:
            logger.critical("Encountered unexpected error while creating %s", env_dir)
            if env_dir.exists():
                logger.warning("Attempting to remove %s", env_dir)
                shutil.rmtree(env_dir, ignore_errors=True)
                msg = "Failed to remove %s" if env_dir.exists() else "Successfully removed %s"
                logger.warning(msg, env_dir)

            raise

    def run_cmd(self, cmd_type, capture_output=False, extra_env=None, synchronous=True, stream_output=False):
        """Run a command.

        :param cmd_type: The command type.
        :type cmd_type: str
        :return:
        """
        try:
            cmdsandargs = cmd_type.split(" ")

            entry_point = self.config["entry_points"][cmdsandargs[0]]["command"]

            # remove the first element,  that is not a file but a command
            args = cmdsandargs[1:]

            # Join entry point and arguments into a single command as a string
            entry_point_args = " ".join(args)
            entry_point = f"{entry_point} {entry_point_args}"

            if self.activate_cmd:
                cmd = _join_commands(self.activate_cmd, entry_point)
            else:
                cmd = _join_commands(entry_point)

            logger.info("Running command: {}".format(cmd))
            _exec_cmd(
                cmd,
                cwd=self.project_dir,
                throw_on_error=True,
                extra_env=extra_env,
                capture_output=capture_output,
                synchronous=synchronous,
                stream_output=stream_output,
            )

            logger.info("Done executing {}".format(cmd_type))
        except IndexError:
            message = "No such argument or configuration to run."
            logger.error(message)
