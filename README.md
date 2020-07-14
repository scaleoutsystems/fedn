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
EXAMPLE=mnist
ALLIANCE_UID=ac435faef-c2df-442e-b349-7f633d3d5523
CLIENT_NAME_BASE=client-fedn1-
MDBUSR=
MDBPWD=
```
(Choose admin username and password for MongoDB)

> you can set EXAMPLE with whatever example you are working on imported with base path from test/your_example
or start all commands below by prepending ```EXAMPLE=mnist``` like ```$ EXAMPLE=data_center docker-compose up```
### Convenience startup
2. a


Build and run all components at once. 
``` 
$ make up
```
_Assumes you have **automake** installed._
### Alternative way
2. To start a bare minimum deployment with one controller, a monitor, Minio, and MongoDB:

````bash 
$ docker-compose up 
````
Navigate to localhost:8081 to see alliance status logs and data in Mongo Express.

3. To attach clients to the controller (can be run on a separate host):
````bash 
$ docker-compose -f mnist-clients.yaml up 
````

4. To attach an orchestrator and start training:
````bash 
$ docker-compose -f combiner.yaml up 
````

5. To enable the Dashboard: 
```bash
docker-compose -f dashboard.yaml up
```
The dashboard can be accessed on localhost:5111 

## Where to go from here? 
Reach out to Scaleout to learn about how FEDn can be deployed in a secure manner together with [STACKn](https://github.com/scaleoutsystems/stackn) to enable ML-alliance governance and life-cycle management of the federated models.  

### License
See LICENSE file.
