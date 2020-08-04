# FEDn

## What is FEDn?
FEDn is an open source framework for Federated Machine Learning (FedML), developed and maintained by Scaleout Systems. 

*Warning, this is an experimental version of the software. Do not use as is for production scenarios!*

<!--- ## Core Features
#### Federated Model Training
#### Multimodal participation
#### Multilevel model combinations
---> 

## Getting started 

The easiest way to start with FEDn is to use the provided docker-compose templates to launch a sandbox environment consisting of the Controller, one Reducer, one Combiner, and a number of Clients. Together, this deploys a minimal system for training a federated model using the Federated Averaging strategy. This repository includes a number of test projects, as specified in this repository in the 'test' folder. These test projects can be used as templates for creating your own model code for the framework. 

Clone the repository and follow these steps: 

1. Create a file named '.env' in the repository root folder and set the following variables:
```yaml
EXAMPLE=mnist-multi-combiner
ALLIANCE_UID=ac435faef-c2df-442e-b349-7f633d3d5523
CLIENT_NAME_BASE=client-fedn1-
MDBUSR=
MDBPWD=
```
(Choose admin username and password for MongoDB)

> you set the EXAMPLE variable to the example you are working on imported with base path from test/your_example
or start all commands below by prepending ```EXAMPLE=mnist``` like ```$ EXAMPLE=data_center docker-compose up```

### Minimal deployment 
1. To deploy the server-side components (Controller, Minio, MongoDB and the Dashboard):

````bash 
$ docker-compose up 
````
Make sure you can reach the controller on 'localhost:8080/controller' before proceeding to the next step. 

2. Attach a combiner:
````bash 
$ docker-compose -f combiner.yaml up 
````

3. Attach a number of clients (assuming you are running the MNIST example):
````bash 
$ docker-compose -f mnist-clients.yaml up 
````

4. Start training

Navigate to localhost:8080/controller, and navigate to the page for the deployed Combiner. There, configure it with the correct UID of a seed model (currently needs to be uploaded separately to Minio), then start training the model using button control.  

## Where to go from here? 
Reach out to Scaleout to learn about how FEDn can be configured and deployed together with [STACKn](https://github.com/scaleoutsystems/stackn) to enable end-to-end ML-alliance governance and life-cycle management of the federated models.  

### License
See LICENSE file.
