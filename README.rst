.. figure:: https://thumb.tildacdn.com/tild6637-3937-4565-b861-386330386132/-/resize/560x/-/format/webp/FEDn_logo.png
   :alt: FEDn logo

.. image:: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml/badge.svg
   :target: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml

.. image:: https://badgen.net/badge/icon/discord?icon=discord&label
   :target: https://discord.gg/KMg4VwszAd

.. image:: https://readthedocs.org/projects/fedn/badge/?version=latest&style=flat
   :target: https://fedn.readthedocs.io

FEDn is a modular and model agnostic framework for hierarchical
federated machine learning which scales from pseudo-distributed
development to real-world production networks in distributed,
heterogeneous environments. For more details see https://arxiv.org/abs/2103.00148.

Core Features
=============

-  **Scalable and resilient.** FEDn is highly scalable and resilient via a tiered 
   architecture where multiple aggregation servers (combiners) form a network to divide up the work to coordinate clients and aggregate models. 
   Recent benchmarks show high performance both for thousands of clients in a cross-device
   setting and for large model updates (1GB) in a cross-silo setting. 
   FEDn has the ability to recover from failure in all critical components.  
   
-  **ML-framework agnostic**. Model updates are treated as black-box
   computations. This means that it is possible to support any
   ML model type or framework. Support for Keras and PyTorch is
   available out-of-the-box.

-  **Security**. A key feature is that
   clients do not have to expose any ingress ports.
 
-  **Track events and training progress**. FEDn logs events in the federation and tracks both training and validation progress in real time. Data is logged as JSON to MongoDB and a user can easily make custom dashboards and visualizations. 

- **UI.** A Flask UI lets users see client model validations in real time, as well as track client training time distributions and key performance metrics for clients and combiners.  

Getting started
===============

Prerequisites
-------------

-  `Docker <https://docs.docker.com/get-docker>`__
-  `Docker Compose <https://docs.docker.com/compose/install>`__
-  `Python 3.8 <https://www.python.org/downloads>`__

Quick start
-----------

The quickest way to get started with FEDn is by trying out the `MNIST
Keras example <https://github.com/scaleoutsystems/fedn/tree/master/examples/mnist-keras>`__. Alternatively, you can start the
base services along with combiner and reducer as it follows.

.. code-block::

   docker-compose up -d

Distributed deployment
======================

We provide instructions for a distributed reference deployment here:
`Distributed
deployment <https://scaleoutsystems.github.io/fedn/deployment.html>`__.

Where to go from here
=====================

-  `Explore additional examples <https://github.com/scaleoutsystems/fedn/tree/master/examples>`__
-  `Understand the
   architecture <https://scaleoutsystems.github.io/fedn/architecture.html>`__
-  `Understand the compute
   package <https://scaleoutsystems.github.io/fedn/tutorial.html>`__

Making contributions
====================

All pull requests will be considered and are much appreciated. Reach out
to one of the maintainers if you are interested in making contributions,
and we will help you find a good first issue to get you started. For
more details please refer to our `contribution
guidelines <https://github.com/scaleoutsystems/fedn/blob/develop/CONTRIBUTING.md>`__.

Documentation
=============
More details about the architecture and implementation:

-  `Documentation <https://fedn.readthedocs.io>`__
-  `Paper <https://arxiv.org/abs/2103.00148>`__

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
