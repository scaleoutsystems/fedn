Welcome to FEDn's documentation!
================================

FEDn is an open-source, modular and model agnostic framework for Federated Machine Learning. Scale seamlessly from pseudo-distributed development to real-world production networks in distributed, heterogeneous environments.

`Scaleout Discord server <https://discord.gg/KMg4VwszAd>`_.

Core Features
=============

- **ML-framework agnostic**. Model updates are treated as black-box computations. This means that it is possible to support virtually any ML model type or framework. Support for Keras and PyTorch is available out-of-the-box, and support for many other model types, including models from SKLearn, are in active development.

- **Horizontally scalable through a tiered aggregation scheme**. FEDn allows for massive horizontal scaling. This is achieved by a tiered architecture where multiple combiners divide up the work to coordinate client updates and aggregation. Recent benchmarks show high performance both for thousands of clients in a cross-device setting and for large model updates (1GB) in a cross-silo setting, see https://arxiv.org/abs/2103.00148.

- **Built for real-world production scenarios**. The implementation is based on proven design patterns in distributed computing and incorporates enterprise security features. A key feature is that data clients do not have to expose any ingress ports.

- **WebUI to manage alliances, track training progress and follow client validations in real time**. The FEDn frontend lets you efficiently manage and track events and training progress in the alliance, helping you monitor both client and server performance.   

Documentation
=============

More details about the architecture and implementation:  

- `Documentation <https://scaleoutsystems.github.io/fedn/>`_.

- `Paper <https://arxiv.org/abs/2103.00148>`_.

Getting started 
===============

The easiest way to start with FEDn is to use the provided docker-compose templates to launch a pseudo-distributed environment consisting of one Reducer, one Combiner, and a few Clients. Together with the supporting storage and database services this makes up a minimal system for developing a federated model and learning the FEDn architecture.  

Clone the repository (make sure to use git-lfs!) and follow these steps:

Pseudo-distributed deployment
-----------------------------

We provide docker-compose templates for a minimal standalone, pseudo-distributed Docker deployment, useful for local testing and development on a single host machine. 

1. Create a default docker network  

We need to make sure that all services deployed on our single host can communicate on the same docker network. Therefore, our provided docker-compose templates use a default external network 'fedn_default'. First, create this network: 

.. code-block:: bash

   docker network create fedn_default

2. Deploy the base services (Minio and MongoDB)  

.. code-block:: bash

   docker-compose -f config/base-services.yaml -f config/private-network.yaml up

Make sure you can access the following services before proceeding to the next steps: 
 
 - `Minio <http://localhost:9000>`_.
 
 - `Mongo Express <http://localhost:8081>`_.
 
3. Start the Reducer  

Copy the settings config file for the reducer, 'config/settings-reducer.yaml.template' to 'config/settings-reducer.yaml'. You do not need to make any changes to this file to run the sandbox. To start the reducer service:

Make sure that you can access the Reducer UI at https://localhost:8090. 

.. code-block:: bash

   docker-compose -f config/reducer.yaml -f config/private-network.yaml up

4. Start a combiner  

Copy the settings config file for the reducer, 'config/settings-combiner.yaml.template' to 'config/settings-combiner.yaml'. You do not need to make any changes to this file to run the sandbox. To start the combiner service and attach it to the reducer:

.. code-block:: bash

   docker network create fedn_defaultdocker-compose -f config/combiner.yaml -f config/private-network.yaml up

Train a federated model
-----------------------

Training a federated model on the FEDn network involves uploading a compute package (containing the code that will be distributed to clients), seeding the federated model with an initial base model (untrained or pre-trained), and then attaching clients to the network. Follow the instruction here to set up the deployed network to train a model for digits classification using the MNIST dataset: 

https://github.com/scaleoutsystems/examples/tree/main/mnist-keras

Fully distributed deployment
============================

The deployment, sizing of nodes, and tuning of a FEDn network in production depends heavily on the use case (cross-silo, cross-device, etc), the size of model updates, on the available infrastructure, and on the strategy to provide end-to-end security. We provide instructions for a fully distributed reference deployment here: [Distributed deployment](https://scaleoutsystems.github.io/fedn/#/deployment). 

Using FEDn in Scaleout Studio 
=============================

Scaleout Studio is a cloud-native SaaS for MLOps for Decentralized AI applications. Studio lets you deploy, manage and monitor FEDn networks as apps deployed to Kubernetes, all from a graphical interface. In addtion to FEDn, Studio provides developer tools (e.g. Jupyter Labs and VSCode), storage managmement (Kubernetes volumes, minio, MongoDB etc), and model serving for the federated model (Tensorflow Serving, TorchServe, MLflow or custom serving). End-to-end example here: https://www.youtube.com/watch?v=-a_nIzkSumI

- Sign up for private-beta access at https://scaleoutsystems.com/.   
- `Deploy STACKn on your own infrastructure <https://github.com/scaleoutsystems/stackn>`_.


Where to go from here
=====================

Explore additional projects/clients:

- PyTorch version of the MNIST getting-started example: https://github.com/scaleoutsystems/examples/tree/main/mnist-pytorch
- Sentiment analysis with a Keras CNN-lstm trained on the IMDB dataset (cross-silo): https://github.com/scaleoutsystems/FEDn-client-imdb-keras 
- Sentiment analysis with a PyTorch CNN trained on the IMDB dataset (cross-silo): https://github.com/scaleoutsystems/FEDn-client-imdb-pytorch.git 
- VGG16 trained on cifar-10 with a PyTorch client (cross-silo): https://github.com/scaleoutsystems/FEDn-client-cifar10-pytorch 
- Human activity recognition with a Keras CNN based on the casa dataset (cross-device): https://github.com/scaleoutsystems/FEDn-client-casa-keras 
- Fraud detection with a Keras auto-encoder (ANN encoder): https://github.com/scaleoutsystems/FEDn-client-fraud_keras  
 
Community support 
=================

Join the `Scaleout Discord Server <https://discord.gg/KMg4VwszAd>`_. to engage with other users and developers. If you have a bug report or a feature request, start a ticket directly here on GitHub. 

Commercial support
==================

Scaleout offers flexible support agreements, reach out at (https://scaleoutsystems.com) to inquire about Enterprise support.

Making contributions
====================

All pull requests will be considered and are much appreciated. Reach out to one of the maintainers if you are interested in making contributions, and we will help you find a good first issue to get you started. 

- `CONTRIBUTING.md <https://github.com/scaleoutsystems/fedn/blob/develop/CONTRIBUTING.md>`_.

Citation
========

If you use FEDn in your research, please cite

.. code-block:: bib

   @article{ekmefjord2021scalable,
   title={Scalable federated machine learning with FEDn},
   author={Ekmefjord, Morgan and Ait-Mlouk, Addi and Alawadi, Sadi and {\AA}kesson, Mattias and Stoyanova, Desislava and Spjuth, Ola and Toor, Salman and Hellander, Andreas},
   journal={arXiv preprint arXiv:2103.00148},
   year={2021}}


Organizational collaborators, contributors and supporters
=========================================================
|pic1| |pic2| |pic3|

.. |pic1| image:: img/logos/Scaleout.png
   :alt: Scaleout logo
   :width: 30%

.. |pic2| image:: img/logos/UU.png
   :alt: Uppsala University logo
   :width: 30%

.. |pic3| image:: img/logos/Scania.png
   :alt: Scania logo
   :width: 30%

License
=======

FEDn is licensed under Apache-2.0 (see LICENSE file for full information).

