.. figure:: https://thumb.tildacdn.com/tild6637-3937-4565-b861-386330386132/-/resize/560x/-/format/webp/FEDn_logo.png
   :alt: FEDn logo

.. image:: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml/badge.svg
   :target: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml

.. image:: https://badgen.net/badge/icon/discord?icon=discord&label
   :target: https://discord.gg/KMg4VwszAd

.. image:: https://readthedocs.org/projects/fedn/badge/?version=latest&style=flat
   :target: https://fedn.readthedocs.io

FEDn enables developers, researchers and data scientists to build federated learning applications that scale from local proof-of-concepts to real-world distributed deployments without code change. 

Core Features
=============

-  **Scalable and resilient.** FEDn enables multiple aggregation servers to share the work to coordinate clients and aggregate models. This makes the framework scalable to large numbers of clients. 
   The system is able to seamlessly recover from failure, enabling robust deployment in production. FEDn also handles asynchronous federated learning scenarios, where clients connect 
   and drop out during training.

-  **Security**. FL clients do not have to open any ingress ports, enabling real-world deployments in a wide range of settigs. Further, FEDn is implemented using secure industry standard communication protocols and 
   supports token-based authentication for FL clients.   

-  **Track events and training progress in real-time**. Extensive event logging and distributed tracing helps developers monitor experiments in real-time, facilitating troubleshooting and auditing.  
   Machine learning validation metrics from clients can be retrieved using the API, enabling flexible analysis of federated experiments. 

-  **ML-framework agnostic**. FEDn is compatible with all major ML frameworks. Examples for Keras, PyTorch and scikit-learn are
   available out-of-the-box.

-  **Deploy your FL project to production on FEDn Studio**. Users can develop a FL use-case in a local development environment, and then deploy it to production on FEDn Studio. FEDn Studio 
   provides the FEDn server-side as a managed service on Kubernetes. A web application provides an intuitive UI for orchestrating runs, visualizing and downloading results, and manage FL client tokens.      



Getting started with the SDK
===============

The best way to get started with the FEDn SDK is to take the quickstart tutorial: 

- `Quickstart PyTorch <https://fedn.readthedocs.io/en/latest/quickstart.html>`__

Documentation
=============
You find more details about the architecture, deployment and how to develop your own application in the documentation:

-  `Documentation <https://fedn.readthedocs.io>`__


Deploying a project to FEDn Studio
===============
Studio provides a managed, production-grade deployment of the FEDn server-side. With Studio you manage token-based authentication for clients, and are able to collaborate with other users in joint project workspaces. In addition to a REST API, Studio has an intuitive Dashboard that let's you manage FL experiments and visualize and download logs and metrics. Follow this guide to `Deploy you project to FEDn Studio <https://guide.scaleoutsystems.com/#/docs>`__ . 


Making contributions
====================

All pull requests will be considered and are much appreciated. For
more details please refer to our `contribution
guidelines <https://github.com/scaleoutsystems/fedn/blob/develop/CONTRIBUTING.md>`__.

Community support
=================

Community support in available in our `Discord
server <https://discord.gg/KMg4VwszAd>`__.

Citation
========

If you use FEDn in your research, please cite:

::

   @article{ekmefjord2021scalable,
     title={Scalable federated machine learning with FEDn},
     author={Ekmefjord, Morgan and Ait-Mlouk, Addi and Alawadi, Sadi and {\AA}kesson, Mattias and Stoyanova, Desislava and Spjuth, Ola and Toor, Salman and Hellander, Andreas},
     journal={arXiv preprint arXiv:2103.00148},
     year={2021}
   }

Organizational collaborators, contributors and supporters
=========================================================

|FEDn logo| |UU logo| |AI Sweden logo| |Zenseact logo| |Scania logo|

License
=======

FEDn is licensed under Apache-2.0 (see `LICENSE <LICENSE>`__ file for
full information).

.. |FEDn logo| image:: https://github.com/scaleoutsystems/fedn/raw/master/docs/img/logos/Scaleout.png
   :width: 15%
.. |UU logo| image:: https://github.com/scaleoutsystems/fedn/raw/master/docs/img/logos/UU.png
   :width: 15%
.. |AI Sweden logo| image:: https://github.com/scaleoutsystems/fedn/raw/master/docs/img/logos/ai-sweden-logo.png
   :width: 15%
.. |Zenseact logo| image:: https://github.com/scaleoutsystems/fedn/raw/master/docs/img/logos/zenseact-logo.png
   :width: 15%
.. |Scania logo| image:: https://github.com/scaleoutsystems/fedn/raw/master/docs/img/logos/Scania.png
   :width: 15%
