![alt text](https://thumb.tildacdn.com/tild6637-3937-4565-b861-386330386132/-/resize/560x/-/format/webp/FEDn_logo.png)
## What is FEDn?
FEDn is an open source framework for Federated Machine Learning (FedML), developed and maintained by Scaleout Systems. 

*Warning, this is an experimental version of the software. Do not use as is for production scenarios!*

## Why use FEDn? 

FEDn provides a flexible framework for building highly scalable federated machine learning systems using the ML framwork of your choice. FEDn is based on open protocols and can be easily integrated with open MLOps frameworks like STACKn to provide end-to-end alliance governance.   

## Core Features
FEDn supports a highly scalable Nested Federated Averaging orchestration scheme. Other protocols, including support for various types of federated ensemble models are in active development. 

<!--- #### Multimodal participation
#### Multilevel model combinations
---> 

## Architecture
Description coming soon. 

## Getting started 

The easiest way to start with FEDn is to use the provided docker-compose templates to launch a sandbox environment consisting of one Reducer, one Combiner, and a number of Clients. Together with the supporting storage and database services (currently Minio and MongoDB), this consitutes a minimal system for training a federated model using the Federated Averaging strategy. This repository bundles a number of test projects. These can be found in the 'test' folder. These projects can be used as templates for creating your own custom federated model using for the framework. 

Clone the repository and follow these steps: 

1. Create a file named '.env' in the repository root folder and set the following variables (alter values as necessary):
```yaml

ALLIANCE_UID=ac435faef-c2df-442e-b349-7f633d3d5523

FEDN_REDUCER_HOST=reducer
FEDN_REDUCER_PORT=8090

FEDN_MONGO_USER=fedn_admin
FEDN_MONGO_PASSWORD=password
FEDN_MONGO_HOST=mongo
FEDN_MONGO_PORT=27017
FEDN_ME_USERNAME=fedn_admin
FEDN_ME_PASSWORD=password

FEDN_MINIO_HOST=minio
FEDN_MINIO_PORT=9000
FEDN_MINIO_ACCESS_KEY=fedn_admin
FEDN_MINIO_SECRET_KEY=password

FEDN_DASHBOARD_HOST=dashboard
FEDN_DASHBOARD_PORT=5111

EXAMPLE=mnist
CLIENT_NAME_BASE=client-fedn1-

```

> you set the EXAMPLE variable to the example you are working on imported with base path from test/your_example
or start all commands below by prepending ```EXAMPLE=mnist``` like ```$ EXAMPLE=data_center docker-compose up```

### Minimal standalone deployment 
We provide templates for a minimal standalone Docker deployment, useful for local testing and development. 

1. To deploy the supporting services (Minio, MongoDB and the Dashboard):

````bash 
$ docker-compose up 
````
Make sure you can access services before proceeding to next steps: 
 - Minio: `localhost:9000` (s3 compatible storage interface)
 - Mongo Express: `localhost:8081` (used for storing model training metadata & statistics)
 - Dashboard: `localhost:5111` (visualizing model training metadata & statistics)

2. Start a Reducer
````bash 
$ docker-compose -f reducer.yaml up 
````

3. Attach a combiner:
````bash 
$ docker-compose -f combiner.yaml up 
````

3. Attach a number of Clients (assuming you are running the MNIST example):
````bash 
$ docker-compose -f mnist-clients.yaml up 
````
Clients will connect as following:
- attach to reducer and ask for assignment
- attach to assigned combiner and ask for work
- if no assignment is available it will ask again after due `timeout`.

> Quicktip: You can ofcourse stack commands for convenience and start all in one by `docker-compose -f A.yaml -f B.yaml up --build`

### Train a federated model

#### Seed the system with an initial model

Navigate to the Minio dashboard and log in. To prepare FEDn to run training, we need to upload a corresponding seed model to the appropriate location in Minio. Creating and staging the seed model is typically done by founding members of the ML alliance. For testing purposes, you find pre-generated seed model in "test/mnist/seed" (and correspondingly for the other examples).  Create a bucker called 'models' and upload the seed model file in that bucket. 

*Note, there is a script "init_model.py" that you can edit if you would like to alter the neural network of the seed model.*

#### Start training
Visiting `https://localhost:8090` if you are running the development docker-compose yamls.
From there you can also start the reducer for basic execution (and set basic parameters).
Alternatively you can start training, just call the Reducer rest API start endpoint: `localhost:8090/start` with curl

if y ou have the mnist example configured you can start the clients by starting the docker mnist configured clients which will.


You can follow the progress of training visually in the Dashboard: 
 - localhost:5111/table 
 - localhost:5111/box

## Where to go from here?
Explore our other example models, or use them as templates to create your own project. 

## Distributed deployment
To deploy the setup across different hosts in a live environment, modify the .env file accordingly for each host/service. Reference deployments are coming soon. 

## Commercial support
Reach out to Scaleout to learn about how FEDn can be customized, configured and deployed to enable production-grade ML-alliances and life-cycle management of the federated models.  

## License
See LICENSE file.
