{
  "name": "devcontainer",
  "dockerFile": "Dockerfile",
  "context": "..",
  "remoteUser": "default",
  // "workspaceFolder": "/fedn",
  // "workspaceMount": "source=/path/to/fedn,target=/fedn,type=bind,consistency=default",
  "extensions": [
    "ms-azuretools.vscode-docker",
    "ms-python.python",
    "exiasr.hadolint",
    "yzhang.markdown-all-in-one",
    "ms-python.isort"
  ],
  "mounts": [
    "source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind,consistency=default",
  ],
  "runArgs": [
    "--net=host"
  ],
  "build": {
    "args": {
      "BASE_IMG": "python:3.9"
    }
  }
}

