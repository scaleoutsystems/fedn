.. figure:: https://thumb.tildacdn.com/tild6637-3937-4565-b861-386330386132/-/resize/560x/-/format/webp/FEDn_logo.png
   :alt: FEDn logo

.. image:: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml/badge.svg
   :target: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml

.. image:: https://badgen.net/badge/icon/discord?icon=discord&label
   :target: https://discord.gg/KMg4VwszAd

.. image:: https://readthedocs.org/projects/fedn/badge/?version=latest&style=flat
   :target: https://fedn.readthedocs.io

FEDn is a highly scalable and resilient framework for
federated machine learning. It let's developers, researchers and data scientists build federated learning applications that scale from local proof-of-concepts to real-world distributed deployments without code change. 

Core Features
=============

-  **Scalable and resilient.** FEDn enables multiple aggregation servers (combiners) to divide up the work to coordinate clients and aggregate models. This makes the framework able to scale to large numbers of clients. 
   The server-side is able to seamlessly recover from failure, making for robust deployment in production scenarios. FEDn is robust in asynchronous federated learning scenarios, seamlessly handling clients that connects 
   and drops out during training.

-  **Security**. FL clients do not have to open any ingress ports. The framework is built using secure industry standard communication protocols and 
   supports token-based authentication for FL clients.   

-  **Track events and training progress in real-time**. Extensive event logging and distributed tracing helps developers monitor experiments in real-time, facilitating troubleshooting and auditing.  
   Tracking and model validation data can easily be retrieved using the API enabling development of custom dashboards and visualizations. 

-  **ML-framework agnostic**. FEDn is compatible with all major ML frameworks. Examples for Keras, PyTorch and scikit-learn are
   available out-of-the-box.

-  **Deploy your FL project to production on FEDn Studio**. Users can develop their FL use-case in a local development environment and then deploy it to production on FEDn Studio. FEDn Studio 
   provides the FEDn server-side as a managed service. A web application provides an intuitive UI for orchestrating runs, visualizing and downloading results, and manage FL client tokens.      



Getting started
===============

The best way to get started with the FEDn SDK is to take the quickstart tutorial: 

- `Quickstart PyTorch <https://fedn.readthedocs.io/en/latest/quickstart.html>`__

Documentation
=============
You find more details about the architecture, deployment and how to develop your own application in the documentation:

-  `Documentation <https://fedn.readthedocs.io>`__


FEDn Studio
===============
You can deploy your FEDn projects to FEDn Studio. Studio provides a managed, production-grade deployment of the FEDn server-side. With Studio you manage token-based authentication for clients, and are able to collaborate with other users in joint project workspaces. In addition to a REST API, Studio has an intuitive Dashboard that let's you manage FL exepriments and visualize and download logs and metrics. Follow this guide to `Deploy you project to FEDn Studio <https://guide.scaleoutsystems.com/#/docs>`__ . 


Making contributions
====================

All pull requests will be considered and are much appreciated. Reach out
to one of the maintainers if you are interested in making contributions,
and we will help you find a good first issue to get you started. For
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
