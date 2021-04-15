![alt text](https://thumb.tildacdn.com/tild6637-3937-4565-b861-386330386132/-/resize/560x/-/format/webp/FEDn_logo.png)
## What is FEDn?
FEDn is an open-source, modular and ML-framework agnostic framework for Federated Machine Learning (FedML) developed and maintained by Scaleout Systems. FEDn enables highly scalable cross-silo and cross-device use-cases over *FEDn networks*.   
  
## Core Features

FEDn lets you seamlessly go from local development and testing in a pseudo-distributed sandbox to live production deployments in distributed, heterogeneous environments. Three key design objectives are guiding the project: 

### A ML-framework agnostic, black-box design
Client model updates and model validations are treated as black-box computations. This means that it is possible to support virtually any ML model type or framework. Support for Keras and PyTorch artificial neural network models are available out-of-the-box, and support for many other model types, including select models from SKLearn, are in active development. A developer follows a structured design pattern to implement clients and there is a lot of flexibility in the toolchains used.  

### Horizontally scalable through a tiered aggregation scheme 
FEDn is designed to allow for flexible and easy scaling to handle growing numbers of clients and to meet latency and throughput requirements spanning cross-silo and cross-device use-cases. This is achieved by a tiered architecture where multiple independent combiners divide up the work to talk to clients and to aggregate local model updates. A reducer protocol then aggregates combiner-level updates into a global model. Recent benchmarks show high performance both for thousands of clients in a cross-device setting and for 40 clients with large model updates (1GB) in a cross-silo setting, see https://arxiv.org/abs/2103.00148. 

### Built for real-world distributed computing scenarios 
FEDn is built groud up to support real-world, production deployments in the distributed cloud. FEDn relies on proven best-practices in distributed computing, uses battle-hardened components, and incorporates enterprise security features. There is no "simulated mode", only distributed mode. However, it is of course possible to run a local sandbox system in pseudo-distributed mode for testing and development.  

## Getting started 

The easiest way to start with FEDn is to use the provided docker-compose templates to launch a local sandbox / simulated environment consisting of one Reducer, two Combiners, and five Clients. Together with the supporting storage and database services (currently Minio and MongoDB), this constitutes a minimal system for training a federated model and learning the FEDn architecture. FEDn projects are templated projects that contain the user-provided model application components needed for federated training. This repository bundles a number of such test projects in the 'test' folder. These projects can be used as templates for creating your own custom federated model. 

Clone the repository (make sure to use git-lfs!) and follow these steps:

### Pseudo-distributed deployment
We provide docker-compose templates for a minimal standalone, pseudo-distributed Docker deployment, useful for local testing and development on a single host machine. 

1. Create a default docker network  

We need to make sure that all services deployed on our single host can communicate on the same docker network. Therefore, our provided docker-compose templates use a default external network 'fedn_default'. First, create this network: 

````bash 
$ docker network create fedn_default
````

2. Deploy the base services (Minio and MongoDB)  

````bash 
$ docker-compose -f config/base-services.yaml up 
````

Make sure you can access the following services before proceeding to the next steps: 
 - Minio: localhost:9000
 - Mongo Express: localhost:8081
 
3. Start the Reducer  

Copy the settings config file for the reducer, 'config/settings-reducer.yaml.template' to 'config/settings-reducer.yaml'. You do not need to make any changes to this file to run the sandbox. To start the reducer service:

````bash 
$ docker-compose -f config/reducer.yaml up 
````

4. Start a combiner  

Copy the settings config file for the reducer, 'config/settings-combiner.yaml.template' to 'config/settings-combiner.yaml'. You do not need to make any changes to this file to run the sandbox. To start the combiner service and attach it to the reducer:

````bash 
$ docker-compose -f config/combiner.yaml up 
````

Make sure that you can access the Reducer UI at https://localhost:8090 and that the combiner is up and running before proceeding to the next step.

### Train a federated model
Training a federated model on the FEDn network involves uploading a compute package, seeding the model, and attaching clients to the network. Follow the instruction here to set the environment up to train a model for digits classification using the MNIST dataset: 

https://github.com/scaleoutsystems/fedn/blob/master/test/mnist-keras/README.md

#### Updating/changing the compute package and/or the seed model
By design, it is not possible to simply delete the compute package to restart the alliance -  this is a security constraint enforced to not allow for arbitrary code package replacement in an already configured federation. To restart and reseed the alliance in development mode navigate to MongoExpress (localhost:8081), log in (credentials are found in the config/base-services.yaml) and delete the entire collection 'fedn-test-network', then restart all services.

## Distributed deployment
The actual deployment, sizing of nodes, and tuning of a FEDn network in production depends heavily on the use case (cross-silo, cross-device, etc), the size of model updates, on the available infrastructure, and on the strategy to provide end-to-end security. You can easily use the provided docker-compose templates to deploy FEDn network across different hosts in a live environment, but note that it might be necessary to modify them slightly depending on your target environment and host configurations.   

This example serves as reference deployment for setting up a fully distributed FEDn network consisting of one host serving the supporting services (Minio, MongoDB), one host serving the reducer, one host running two combiners, and one host running a variable number of clients. 

> Warning, there are additional security considerations when deploying a live FEDn network, outside of core FEDn functionality. Make sure to include these aspects in your deployment plans.

### Prerequisite for the reference deployment

#### Hosts
This example assumes root access to 4 Ubuntu 20.04 Servers for running the FEDn network. We recommend at least 4 CPU, 8GB RAM flavors for the base services and the reducer, and 4 CPU, 16BG RAM for the combiner host. Client host sizing depends on the number of clients you plan to run. You need to be able to configure security groups/ingress settings for the service node, combiner, and reducer host.

#### Certificates
Certificates are needed for the reducer and combiner services. By default, FEDn will generate unsigned certificates for the reducer and combiner nodes using OpenSSL. 

> Certificates based on IP addresses are not supported due to incompatibilities with gRPC. 

### 1. Deploy supporting services  
First, deploy Minio and Mongo services on one host (make sure to change the default passwords). Confirm that you can access MongoDB via the MongoExpress dashboard before proceeding with the reducer.  

### 2. Deploy the reducer
Follow the steps for pseudo-distributed deployment, but now edit the settings-reducer.yaml file to provide the appropriate connection settings for MongoDB and Minio from Step 1. Also, copy 'config/extra-hosts-reducer.yaml.template' to 'config/extra-hosts-reducer.yaml' and edit it, adding a host:IP mapping for each combiner you plan to deploy. Then you can start the reducer: 

```bash
sudo docker-compose -f config/reducer.yaml -f config/extra-hosts-reducer.yaml up 
```

### 3. Deploy combiners
Edit 'config/settings-combiner.yaml' to provide a name for the combiner (used as a unique identifier for the combiner in the network), a hostname (which is used by reducer and clients to connect to combiner RPC), and the port (default is 12080, make sure to allow access to this port in your security group/firewall settings). Also, provide the IP and port for the reducer under the 'controller' tag. Then deploy the combiner: 

```bash
sudo docker-compose -f config/combiner.yaml up 
```

Optional: Repeat the same steps for the second combiner node. Make sure to provide unique names for the two combiners. 

> Note that it is not currently possible to use the host's IP address as 'host'. This is due to gRPC not being able to handle certificates based on IP. 

### 4. Attach clients to the FEDn network
Once the FEDn network is deployed, you can attach clients to it in the same way as for the pseudo-distributed deployment. You need to provide clients with DNS information for all combiner nodes in the network. For example, to start 5 unique MNIST clients on a single host, copy  'config/extra-hosts-clients.template.yaml' to 'test/mnist-keras/extra-hosts.yaml' and edit it to provide host:IP mappings for the combiners in the network. Then, from 'test/mnist-keras':

```bash
sudo docker-compose -f docker-compose.yaml -f config/extra-hosts-client.yaml up --scale client=5 
```

## Using FEDn from the STACKn SaaS
STACKn, Scaleout's SaaS for MLOps in distributed cloud, has experimental UI functionality for deploying and testing a FEDn network as 'Apps' directly from the UI, as well as one-click Apps for serving the federated model using e.g. Tensorflow Serving, TorchServe, MLflow or custom serving. Refer to the STACKn documentation to set this up, or reach out to Scaleout for a demo/access to a pre-alpha SaaS deployment.   

## Where to go from here
Additional example projects/clients:

- PyTorch version of the MNIST getting-started example in test/mnist-pytorch
- Sentiment analyis with a Keras CNN-lstm trained on the IMDB dataset (cross-silo): https://github.com/scaleoutsystems/FEDn-client-imdb-keras 
- Sentiment analyis with a PyTorch CNN trained on the IMDB dataset (cross-silo): https://github.com/scaleoutsystems/FEDn-client-imdb-pytorch.git 
- VGG16 trained on cifar-10 with a PyTorch client (cross-silo): https://github.com/scaleoutsystems/FEDn-client-cifar10-pytorch 
- Human activity recognition with a Keras CNN based on the casa dataset (cross-device): https://github.com/scaleoutsystems/FEDn-client-casa-keras 
 
## Support
For more details please check out the FEDn documentation (https://scaleoutsystems.github.io/fedn/). If you don't find the information that you're looking for, please reach out to Scaleout (https://scaleoutsystems.com) or start a ticket directly here on GitHub.

## Contributions
All pull requests will be considered and are much appreciated. We are currently managing issues in an external tracker (Jira). Reach out to one of the maintainers if you are interested in making contributions, and we will help you find a good first issue to get you started. 

For development, it is convenient to use the docker-compose templates config/reducer-dev.yaml and config/combiner-dev.yaml. These files will let you conveniently rebuild the reducer and combiner images with the current local version of the fedn source tree instead of the latest stable release. You might also want to use a Dockerfile for the client that installs fedn from your local clone of FEDn, alternatively mounts the source. 

## License
FEDn is licensed under Apache-2.0 (see LICENSE file for full information).
