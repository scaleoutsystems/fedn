![alt text](https://thumb.tildacdn.com/tild6637-3937-4565-b861-386330386132/-/resize/560x/-/format/webp/FEDn_logo.png)
## What is FEDn?
FEDn is an open source, modular framework for Federated Machine Learning (FedML), developed and maintained by Scaleout Systems. It enables developers to configure and deploy *FEDn networks* to support different federated scenarios, ranging from cross-silo to cross-device use-cases.   
  
## Core Features

Three key design objectives are guiding development in the project and is reflected in the core features: 

### Horizontally scalable through a tiered aggregation scheme 
FEDn is designed to allow for flexible and easy scaling to meet both the demands from a growing number of clients, and from latency and throughput requirements spanning cross-silo and cross-device cases. This is addressed by allowing for a tiered model update and model aggregation scheme where multiple combiners divide up the work for global aggregation steps.  

### A ML-framework agnostic, black-box design
The framework treats client model updates and model validations as black-boxes. A developer can follow a structured design pattern to implement a custom helper class to support any ML model type or framework. Support for Keras Sequential models are available out-of-the box, and support for the TF functional API, PyTorch and SKLearn are in active development.  

### Built for real-world distributed computing scenarios 
FEDn is built to support real-world, production deployments. FEDn relies on proven best-practices in distributed computing, uses battle-hardened components, and incorporates enterprise security features. There is no "simulated mode", only distributed mode. However, it is of course possible to run a local sandbox system in pseudo-distributed mode for convenient testing and devepment.  

## Architecture

Constructing a federated model with FEDn amounts to a) specifying the details of the client side training code and data integrations, and b) deploying the  reducer-combiner network. A FEDn network, as illustrated in the picture below, is made up of three main components: the *Reducer*, one or more *Combiners* and a number of *Clients*. The combiner network forms the backbone of the FedML orchestration mechanism, while the Reducer provides discovery services and provides controls to coordinate training over the combiner network. By horizontally scaling the combiner network, one can meet the needs from a growing number of clients.  
 
![alt-text](https://github.com/scaleoutsystems/fedn/blob/update-readme/docs/img/overview.png)

### Main components

#### Client
A Client is a data node, holding private data and connecting to a Combiner to recieve model update requests and model validation requests during trainig rounds. Importantly, clients do not require any open ingress ports. A client recieves the code to be executed from the Reducer upon connecting to the network, and thus they only need to be configured prior to connection to read the local datasets during training and validation. A Python3 client implementation is provided out of the box, and it is possible to write clients in a variery of languages to target different software and hardware requirements.  

#### Combiner
A combiner is an actor which main role is to orchestrate and aggregat model updates from a number clients during a training round. When and how to trigger such orchestration rounds are specified in the overall *compute plan* laid out by the Reducer. Each combiner in the network runs an independent gRPC server, providing RPCs for interacting with the alliance subsystem it controls. Hence, the total number of clients that can be accomodated in a FEDn network is proportional to the number of active combiners in the FEDn network. Combiners can be deployed anywhere, e.g. in a cloud or on a fog node to provide aggregation services near the cloud edge. 

#### Reducer
The reducer fills three main roles in the FEDn network: 1.) it lays out the overall, global training strategy and comminicates that to the combiner network. It also dictates the strategy to aggregate model updates from individual combiners into a single global model, 2.) it handles global state and maintains the *model trail* - an immutable trail of global model updates uniquely defining the FedML training timeline, and  3.) it provides discovery services, mediating connections between clients and combiners. For this purpose, the Reducer exposes a standard REST API. 

### Services and communication
The figure below provides a logical archiecture view of the services provided by each agent and how they interact. 

![Alt text](docs/img/FEDn-arch-overview.png?raw=true "FEDn architecture overview")

### Control flows and algorithms
FEDn is desinged to allow customization of the FedML algorithm, following a specified pattern, or programming model. Model aggregation happens on two levels in the system. First, each Combiner can be configured with a custom orchestration and aggregation implementation, that reduces model updates from Clients into a single, *combiner level* model. Then, a configurable aggregation protocol on Reducer level is responsible for combining the combiner-level models into a global model. By varying the aggregation schemes on the two levels in the system, many different possible outcomes can be achieved. Good staring configurations are provided out-of-the box to help the user get started. 

#### Hierarachical Federated Averaging
The currently implemented default scheme uses a local SGD strategy on the Combiner level aggregation, and a simple average of models on the reducer level. This results in a highly horizontally scalable FedAvg scheme. The strategy works well with most artificial neural network (ANNs) models, and can in general be applied to  models where it is possible and makes sense to form mean values of model parameters (for example SVMs). Additional FedML training protocols, including support for various types of federated ensemble models, are in active development.  
![Alt text](docs/img/HFedAvg.png?raw=true "FEDn architecture overview")


## Getting started 

The easiest way to start with FEDn is to use the provided docker-compose templates to launch a local sandbox / simulated environment consisting of one Reducer, two Combiners, and five Clients. Together with the supporting storage and database services (currently Minio and MongoDB), this consitutes a minimal system for training a federated model and learning the FEDn architecture. FEDn projects are templated projects that contain the user-provided model appplication components needed for federated training. This repository bundles a number of such test projects in the 'test' folder. These projects can be used as templates for creating your own custom federated model. 

Clone the repository (make sure to use git-lfs!) and follow these steps:

### Pseudo-distributed deployment 
We provide templates for a minimal standalone, pseudo-distributed Docker deployment, useful for local testing and development. 

1. To deploy the supporting services (Minio and MongoDB):

````bash 
$ docker-compose up 
````
Make sure you can access the following services before proceeding to next steps: 
 - Minio: localhost:9000
 - Mongo Express: localhost:8081
 
2. Start a Reducer

Copy the settings config file for the reducer, 'config/settings-reducer.yaml.template' to 'config/settings-reducer.yaml'. You do not need to make any changes to this file to run the sandbox. To start the reducer service:

````bash 
$ EXAMPLE=mnist docker-compose -f reducer.yaml up 
````

> You set the EXAMPLE variable to the example you are working on imported with base path from test/your_example. 

3. Start a combiner:
Copy the settings config file for the reducer, 'config/settings-combiner.yaml.template' to 'config/settings-combiner.yaml'. You do not need to make any changes to this file to run the sandbox. To start the combiner service and attach it to the reducer:

````bash 
$ docker-compose -f combiner.yaml up 
````

3. Attach two Clients:
Copy the settings config file for the reducer, 'config/settings-client.yaml.template' to 'config/settings-client.yaml'. You do not need to make any changes to this file to run the sandbox. To start the combiner service and attach it to the reducer:

````bash 
$ EXAMPLE=mnist docker-compose -f client.yaml up --scale client=2
````

Make sure that you can access the Reducer UI at https://localhost:8090, and that the combiner and clients are up and running, before proceeding to the next step.

### Train a federated model

#### Seed the system with a base model

To prepare FEDn to run training, we need to upload a seed model via this endpoint (https://localhost:8090/history). Creating and staging the seed model is typically done by the founding members of the ML alliance. For testing purposes, you find a pre-generated seed model in "test/mnist/seed" (and correspondingly for the other examples).

> There is a script "test/mnist/seed/init_model.py" that you can edit if you want to alter the neural network achitecture of the seed model.

#### Start training the model
To start training the model, navigate to the Reducer REST API endpoint: localhost:8090/start.  You can follow the progress of training visually at https://localhost:8090/plot. 
 
## Distributed deployment
The actual deployment, sizing of nodes, and tuning of a FEDn network in production depends heavily on the use case (cross-silo, cross-device etc), the size of model updates, on the available infrastructure, and on the strategy to provide end-to-end security. To deploy a FEDn network across different hosts in a live environment, first analyze the use case and create an appropriate deployment/architecture plan.   

> Warning, there are additional security considerations when deploying a live FEDn network, outside of core FEDn functionality. Make sure to include these aspects in your deployment plans.

This example serves as reference deployment for setting up a fully distributed FEDn network consisting of one host serving the supporting services (Minio, MongoDB), one host serving the reducer, one host running two combiners, and one host running a variable number of clients. 

### Prerequisite for the reference deployment

#### Hosts
This example assumes root access to 4 Ubuntu 20.04 hosts for the FEDn network. We recommend at least 4 CPU, 8GB RAM flavors for the base services and the reducer, and 4 CPU, 16BG RAM for the combiner host. Client host sizing depends on the number of clients you plan to run. You need to be able to configure security groups / ingress settings for the service node, combiner and reducer host.

#### Certificates
Certificates are needed for the reducer and combiner services. By default, FEDn will generate unsigned certificates for the reducer and combiner nodes using OpenSSL. 

> Certificates based on IP addresses are not supported due to incompatibilities with gRPC. 

### 1. Deploy supporting services  
First deploy Minio and Mongo services. Edit the config files 'config/minio.env', 'config/mongodb.env' and 'config/mongoexpress.env' according to your setup. Make sure to change the default passwords. The deploy as in the above example. Confirm that you can access MongoDB via the MongoExpress dashboard before proceeding with the reducer.  

> Skip this step if you already have API access to Minio and MongoDB services. 

### 2. Deploy the reducer
Follow the steps for pseudo-distributed deployment, but now edit the settings-reducer.yaml file to provide the appropriate connection settings for MongoDB and Minio. Also, copy 'config/extra-hosts-reducer.yaml.template' to 'config/extra-hosts-reducer.yaml' and edit it to provide mappings from the 'host' parameter in the combiner configuration. The you can start the reducer:  s

```bash
EXAMPLE=mnist sudo docker-compose -f reducer.yaml -f config/extra-hosts-reducer.yaml up 
```

### 3. Deploy combiners
Edit 'config/settings-combiner.yaml' to provide a name for the combiner (used as a unique identifier for the combiner in the network), a host name (which is used by reducer and clients to connect to combiner RPC) and the port. Also provide connection information to the reducer under 'controller'. Then deploy the combiner: 

```bash
sudo docker-compose -f combiner.yaml up 
```

Repeate the same step for the second combiner node. Make sure to provide unique names for the two combiners. 

> Note that is is not currently possible to use the node IP as 'host'. This is due to gRPC not being able to handle certificates based on IP. 

### 4. Attach clients to the FEDn network
Once the FEDn network is deployed, you can attach clients to it in the same way as for the pseudo-distributed deployment. You need to provide clients with DNS information for all combiner nodes in the network, via 'config/extra-hosts-clients.yaml'. For example, to start 5 unique MNIST clients on a host: 

```bash
EXAMPLE=mnist sudo docker-compose -f client.yaml -f config/extra-hosts-client.yaml up --scale client=5 
```
 
## Support
Reach out to Scaleout (https://scaleoutsystems.com) to learn how to configure and deploy zero-trust FEDn networks in production based on FEDn, and how to adapt FEDn to support a range of use-case scenarios.

## Contributions
All pull requests will be considered. We are currently managing issues in an external tracker (Jira). Reach out to one of the maintainers if you are interested in making contributions, and we will help you find a good first issue to get started. 

## License
FEDn is licensed under Apache-2.0 (see LICENSE file for full information).
