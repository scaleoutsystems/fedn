# MNIST Quickstart (PyTorch version)
This classic example of hand-written text recognition is well suited both as a lightweight test when learning FEDn and when developing on FEDn in pseudo-distributed mode. A normal high-end laptop or a workstation should be able to sustain a few clients. 

## Table of Contents
- [MNIST Example (PyTorch version)](#mnist-example-pytorch-version)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Running the example (pseudo-distributed)](#running-the-example-pseudo-distributed)
  - [Clean up](#clean-up)
  - [Connecting to a distributed deployment](#connecting-to-a-distributed-deployment)

## Prerequisites
- [Ubuntu 20.04, 21.04 or 22.04](https://releases.ubuntu.com/20.04) or [macOS 11](https://apps.apple.com/us/app/macos-big-sur)
- [Docker](https://docs.docker.com/get-docker)
- [Docker Compose](https://docs.docker.com/compose/install)
- [Python 3.8, 3.9 or 3.10](https://www.python.org/downloads)

## Running the example (pseudo-distributed, single host)

### Preparing the environment, the local data, the compute package and seed model
Start by initializing a virtual enviroment with all of the required dependencies.
```
bin/init_venv.sh
```

Then, to get the data you can run the following script.
```
bin/get_data
```

The next command splits the data in 2 parts for the clients.
```
bin/split_data
```
> **Note**: run with `--n_splits=N` to split in *N* parts.

Create the compute package and a seed model that you will be asked to upload in the next step.
```
bin/build.sh
```
> The files location will be `package/package.tgz` and `seed.npz`.

### Deploy FEDn 
Now we are ready to deploy FEDn with `docker-compose`.
```
docker-compose -f ../../docker-compose.yaml up minio mongo mongo-express reducer combiner -d
```

### Initialize the federated model 
Now navigate to http://localhost:8090 to see the reducer UI. You will be asked to upload the compute package and the seed model that you created in the previous step. Make sure to choose the "PyTorch" helper.

### Attach clients 
To attach clients to the network, start by downloading the client configuration file, `client.yaml`. You can either navigate to http://localhost:8090/network and download it via the UI, or run the following command.

```bash
curl -k http://localhost:8090/config/download > client.yaml
```

Now we are ready to start and connect clients: 
```
docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up client
```
> **Note**: run with `--scale client=N` to start *N* clients.

### Run federated training 
Finally, you can start the experiment from the "control" tab of the UI.

## Clean up
You can clean up by running `docker-compose down`.

## Connecting a client to a distributed deployment
To start and remotely connect a client with the required dependencies for this example, start by downloading the `client.yaml` file. You can either navigate to the reducer UI or run the following command:

```bash
curl -k https://<reducer-fqdn>:<reducer-port>/config/download > client.yaml
```
> **Note** make sure to replace `<reducer-fqdn>` and `<reducer-port>` with appropriate values.

Now you are ready to start the client via Docker by running the following command.

```bash
docker run -d \
  -v $PWD/client.yaml:/app/client.yaml \
  -v $PWD/data:/var/data \
  -e ENTRYPOINT_OPTS=--data_path=/var/data/mnist.npz \
  ghcr.io/scaleoutsystems/fedn/fedn:develop-mnist-pytorch run client -in client.yaml
```
> **Note** If reducer and combiner host names, as specfied in the configuration files, are not resolvable in the client host network you need to use the docker option `--add-hosts` to make them resolvable. Please refer to the Docker documentation for more detail.
