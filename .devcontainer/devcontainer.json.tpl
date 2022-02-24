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
    "yzhang.markdown-all-in-one"
  ],
  "mounts": [
    "source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind,consistency=default",
  ],
  "forwardPorts": [
    8090,
    9000,
    9001,
    8081
  ],
  "runArgs": [
    "--net=host"
  ],
}
