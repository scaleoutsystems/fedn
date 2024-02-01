.. figure:: https://thumb.tildacdn.com/tild6637-3937-4565-b861-386330386132/-/resize/560x/-/format/webp/FEDn_logo.png
   :alt: FEDn logo

.. image:: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml/badge.svg
   :target: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml

.. image:: https://badgen.net/badge/icon/discord?icon=discord&label
   :target: https://discord.gg/KMg4VwszAd

.. image:: https://readthedocs.org/projects/fedn/badge/?version=latest&style=flat
   :target: https://fedn.readthedocs.io

FEDn is a modular and model agnostic framework for
federated machine learning. FEDn is designed to scale from pseudo-distributed
development on your laptop to real-world production setups in geographically distributed environments. 

Core Features
=============

-  **Scalable and resilient.** FEDn is scalable and resilient via a tiered 
   architecture where multiple aggregation servers (combiners) divide up the work to coordinate clients and aggregate models. 
   Benchmarks show high performance both for thousands of clients in a cross-device
   setting and for large model updates in a cross-silo setting. 
   FEDn has the ability to recover from failure in all critical components. 

-  **Security**. FEDn is built using secure industry standard communication protocols (gRPC). A key feature is that
   clients do not have to expose any ingress ports. 

-  **Track events and training progress in real-time**. FEDn tracks events for clients and aggregation servers, logging to MongoDB. This
   helps developers monitor traning progress in real-time, and to troubleshoot the distributed computation.  
   Tracking and model validation data can easily be retrieved using the API enabling development of custom dashboards and visualizations. 

-  **Flexible handling of asynchronous clients**. FEDn supports flexible experimentation 
   with clients coming in and dropping out during training sessions. Extend aggregators to experiment 
   with different strategies to handle so called stragglers.

-  **ML-framework agnostic**. Model updates are treated as black-box
   computations. This means that it is possible to support any
   ML model type or framework. Support for Keras and PyTorch is
   available out-of-the-box.


Getting started
===============

The best way to get started is to take the quickstart tutorial: 

- `Quickstart PyTorch <https://fedn.readthedocs.io>`__

Documentation
=============
You will find more details about the architecture, compute package and how to deploy FEDn fully distributed in the documentation:

-  `Documentation <https://fedn.readthedocs.io>`__
-  `Paper <https://arxiv.org/abs/2103.00148>`__


FEDn Studio
===============
Scaleout develops a Django Application, FEDn Studio, that provides a UI, authentication/authorization, client identity management, project-based multitenancy for manging multiple projects, and integration with your MLOps pipelines.
There are also additional tooling and charts for deployments on Kubernetes including integration with several projects from the cloud native landscape. See  `FEDn Framework <https://www.scaleoutsystems.com/framework>`__ 
for more information. 


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
