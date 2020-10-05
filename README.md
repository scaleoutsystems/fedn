![alt text](https://thumb.tildacdn.com/tild6637-3937-4565-b861-386330386132/-/resize/560x/-/format/webp/FEDn_logo.png)
## What is FEDn?
FEDn is an open source framework for Federated Machine Learning (FedML), developed and maintained by Scaleout Systems.

FEDn is modular and enables developers to configure and deploy FedML networks for different use-cases and deployment scenarios, ranging from cross-silo to cross-device. The framework takes a ML-framework agnostic approach to training federated models. 

*Warning, this is an early version of the software. Do not use as is for production scenarios!*

## Core Features
FEDn currently supports a highly horizontally scalable Hierarchical Federated Averaging orchestration scheme.  The present version supports Keras Sequential models out of the box, but a user can implement a custom helper class to support clients based on other ML frameworks. Other FedML training protocols, including support for various types of federated ensemble models, and helpers for PyTorch (as well as other popular frameworks), are in active development. 

## Architecture

A FEDn network, as illustrated in the picture below, is made up of three key agents: a *Reducer*, one or more *Combiners* and a number of *Clients*. 

#### Client
A Client holds private data and connects to a Combiner to recieve model update requests and model validation requests. Clients need to be configured to be able to execute model training for the ML-model type used.  

#### Combiner
A combiner is an actor which orchestrates and aggregates model updates from a number of attached clients. When and how to trigger such orchestration rounds are specified in *compute plans* sent by the Reducer. Each combiner in the network is an independent (and identical) gRPC Server, providing RPCs for interacting with its own alliance subsystem. Hence, the total number of clients that can be accomodated in a network is proportional to the number of active combiners. 

#### Reducer
The reducer fills three main roles in the network: 1.) To lay out and initialize the overall, global training strategy and to aggregate model updates from Combiners into a global model, 2.) to handle global state and maintain the model ledger - an immutable trail of global model updates uniquely defining FedML timeline, and  3) act as a discoverey service, mediating connections between Clients and Combiners. For this purpose, the Reducer exposes a standard REST API.  

![alt-text](https://github.com/scaleoutsystems/fedn/blob/update-readme/docs/img/overview.png)

Training a federated model with FEDn amounts to specifying the Client side code and planning the deployment of the above components to meet the practical requirements from the use case at hand. 

## Algorithms
FEDn is desinged to allow customization of the FedML algorithm, following a specified pattern, or programming model. Model aggregation happens on two levels in the system. First, each Combiner can be configured with a custom orchestration and aggregation implementation, that reduces model updates from Clients into a single, *combiner level* model. Then, a configurable aggregation protocol on Reducer level is responsible from combining the combiner-level models into a global model. By varying the aggregation scheme on the two levels in the system, many different possible outcomes can be achieved. 

#### Hierarachical Federated Averaging
The currently implemented default scheme uses the FedAvg strategy on the Combiner level aggregation, and a simple average of models on the reducer level. This results in a highly horizontally scalable scheme. The strategy works well with most artificial neural network (ANNs) models, and can in general be applied to  models where it is possible / makes sense to form mean values of model parameters (e.g. support vector machines). 

## Getting started 

The easiest way to start with FEDn is to use the provided docker-compose templates to launch a local sandbox environment consisting of one Reducer, two Combiners, and five Clients. Together with the supporting storage and database services (currently Minio and MongoDB), this consitutes a minimal system for training a federated model using the Federated Averaging strategy. FEDn projects are templated projects that contain the user-provided model appplication components needed for federated training. This repository bundles a number of such test projects in the 'test' folder. These projects can be used as templates for creating your own custom federated model using the framework. 

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

FEDN_DASHBOARD_HOST=localhost
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
Make sure you can access the following services before proceeding to next steps: 
 - Minio: localhost:9000
 - Mongo Express: localhost:8081
 - Dashboard: localhost:5111
 
2. Start a Reducer
````bash 
$ docker-compose -f reducer.yaml up 
````

3. Attach two combiners:
````bash 
$ docker-compose -f combiner.yaml up 
````

3. Attach a number of Clients (assuming you are running the MNIST example):
````bash 
$ docker-compose -f mnist-clients.yaml up 
````

Make sure that you can access the Reducer UI at https://localhost:8090, and that the combiner and clients are up and running, before proceeding to the next step.

### Train a federated model

#### Seed the system with an initial model

Navigate to the Minio dashboard and log in. To prepare FEDn to run training, we need to upload a seed model to the appropriate location in Minio. Creating and staging the seed model is typically done by founding members of the ML alliance. For testing purposes, you find pre-generated seed model in "test/mnist/seed" (and correspondingly for the other examples).  Create a bucket called 'models' and upload the seed model file there. 

*Note, there is a script "init_model.py" that you can edit if you would like to alter the actual structure of the seed model.*

#### Start training
To start training, navigate to the Reducer REST API endpoint: localhost:8090/start 

You can follow the progress of training visually in the Dashboard: 

 - localhost:5111/table 
 - localhost:5111/box

## Distributed deployment

The actual deployment, sizing and tuning of a FEDn network in production depends heavily on the use case (cross-silo, cross-device etc), the size of models and on the available infrastructure. To deploy a setup across different hosts in a live environment, create an architecture plan and modify the .env file accordingly for each host/service. You also need provide signed certificates for the various services. Reference deployment descriptions for representative scenarios and hardware are coming soon. 

## Where to go from here?
Explore our other example models, or use them as templates to create your own project. 

## Support
Reach out to Scaleout (https://scaleoutsystems.com) to learn about how FEDn can be customized, configured and deployed to enable production-grade ML-alliances and alliance life-cycle management and governance for federated models.  

## Contributions
All pull requests will be considered. We are currently managing issues in an external tracker (Jira). Reach out to one of the maintainers if you are interested in making contributions, and we will help you find a good first issue to get started. 

## License
FEDn is licenced under Apache-2.0 (see LICENSE file for full information).
