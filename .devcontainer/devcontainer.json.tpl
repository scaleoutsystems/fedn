{
  "name": "devcontainer",
  "dockerFile": "Dockerfile",
  "context": "..",
  "remoteUser": "default",
  // "workspaceFolder": "/fedn",
  // "workspaceMount": "source=/path/to/fedn,target=/fedn,type=bind,consistency=default",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-azuretools.vscode-docker",
        "ms-python.python",
        "exiasr.hadolint",
        "yzhang.markdown-all-in-one",
        "charliermarsh.ruff"
      ]
    }
  },
  "mounts": [
    "source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind,consistency=default"
  ],
  "runArgs": [
    "--net=host"
  ],
  "build": {
    "args": {
      "BASE_IMG": "python:3.11"
    }
  }
}