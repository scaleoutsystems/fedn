# MNIST Example (PyTorch version)
This classic example of hand-written text recognition is well suited both as a lightweight test when learning FEDn and developing on FEDn in psedo-distributed mode. A normal high-end laptop or a workstation should be able to sustain a few clients. 

## Table of Contents
- [MNIST Example (PyTorch version)](#mnist-example-pytorch-version)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Running the example](#running-the-example)
  - [Clean up](#clean-up)

## Prerequisites
- [Docker](https://docs.docker.com/get-docker)
- [Docker Compose](https://docs.docker.com/compose/install)
- [Python 3.8](https://www.python.org/downloads)

## Running the example
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


Now we are ready to start the pseudo-distributed deployment with `docker-compose`.
```
docker-compose -f ../../docker-compose.yaml -f docker-compose.overide up -d
```
> **Note**: run with `--scale client=N` to start *N* clients.

Now navigate to https://localhost:8090 to see the reducer UI. You will be asked to upload a compute package and a seed model that you can generate by running the following script.
```
bin/build.sh
```
> The files location will be `package/package.tgz` and `seed.npz`.

Finally, you can start the experiment from the "control" tab.

## Clean up
You can clean up by running `docker-compose down`.