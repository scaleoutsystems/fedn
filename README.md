# FEDn

## What is FEDn?
FEDn is an open source collaborative AI platform for federated machine learning.

*Warning, this is an experimental version of the software. Do not use as is for production scenarios!*

## Core Features
#### Federated Model Training
#### Multimodal participation
#### Multilevel model combinations

## Getting started 

The easiest way to start developing on FEDn is to use docker-compose to launch a sandbox environment. with one controller, one monitor, one client and the FedAvg orchestrator. Test projects that can be deployed are specified in this repository in the 'test' folder. 

1. Create a .env file and set the following variables.
```yaml
EXAMPLE=mnist-multi-combiner
ALLIANCE_UID=ac435faef-c2df-442e-b349-7f633d3d5523
CLIENT_NAME_BASE=client-fedn1-
MDBUSR=
MDBPWD=
```
(Choose admin username and password for MongoDB)

> you can set EXAMPLE with whatever example you are working on imported with base path from test/your_example
or start all commands below by prepending ```EXAMPLE=mnist``` like ```$ EXAMPLE=data_center docker-compose up```

### Deploy a minimal set of components
1. To start a bare minimum deployment with one controller, a monitor, Minio, MongoDB and the Dashboard:

````bash 
$ docker-compose up 
````

2. Attach a combiner:
````bash 
$ docker-compose -f combiner.yaml up 
````

3. Attach clients (assuming you are running the MNIST example):
````bash 
$ docker-compose -f mnist-clients.yaml up 
````

Navigate to localhost:8080/controller to see an overview of the alliance configuration and to start training the model.  


## Where to go from here? 
Reach out to Scaleout to learn about how FEDn can be deployed in a secure manner together with [STACKn](https://github.com/scaleoutsystems/stackn) to enable ML-alliance governance and life-cycle management of the federated models.  

### License
See LICENSE file.
