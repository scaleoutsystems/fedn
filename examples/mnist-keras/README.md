# MNIST (TensorFlow/Keras version)
This classic example of hand-written text recognition is well suited both as a lightweight test when developing on FEDn in pseudo-distributed mode. A normal high-end laptop or a workstation should be able to sustain a few clients. The example automates the partitioning of data and deployment of a variable number of clients on a single host. We here assume working experience with containers, Docker and docker-compose. 

## Table of Contents
- [MNIST Example (Keras version)](#mnist-example-keras-version)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Running the example (pseudo-distributed)](#running-the-example-pseudo-distributed)
  - [Clean up](#clean-up)

## Prerequisites
- [Python 3.8, 3.9 or 3.10](https://www.python.org/downloads)
- [Docker](https://docs.docker.com/get-docker)
- [Docker Compose](https://docs.docker.com/compose/install)

## Running the example (pseudo-distributed)
Clone FEDn and locate into this directory.
```sh
git clone https://github.com/scaleoutsystems/fedn.git
cd fedn/examples/mnist-keras
```

### Preparing the environment, the local data, the compute package and seed model

Start by initializing a virtual enviroment with all of the required dependencies.
```sh
bin/init_venv.sh
```

Then, to get the data you can run the following script.
```sh
bin/get_data
```

The next command splits the data in 2 parts for the clients.
```sh
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
docker-compose -f ../../docker-compose.yaml up -d minio mongo mongo-express reducer combiner
```

### Initialize the federated model 
Now navigate to http://localhost:8090 to see the reducer UI. You will be asked to upload the compute package and the seed model that you created in the previous step.

### Attach clients 
To attach clients to the network, use the docker-compose.override.yaml template to start 2 clients: 

```
docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up client 
```
> **Note**: run with `--scale client=N` to start *N* clients.

### Run federated training 
Finally, you can start the experiment from the "control" tab of the UI. 

## Clean up
You can clean up by running `docker-compose down`.
