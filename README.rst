.. figure:: https://thumb.tildacdn.com/tild6637-3937-4565-b861-386330386132/-/resize/560x/-/format/webp/FEDn_logo.png
   :alt: FEDn logo

.. image:: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml/badge.svg
   :target: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml

.. image:: https://badgen.net/badge/icon/discord?icon=discord&label
   :target: https://discord.gg/KMg4VwszAd

FEDn is a modular and model agnostic framework for hierarchical
federated machine learning which scales from pseudo-distributed
development to real-world production networks in distributed,
heterogeneous environments.

Core Features
=============

-  **ML-framework agnostic**. Model updates are treated as black-box
   computations. This means that it is possible to support virtually any
   ML model type or framework. Support for Keras and PyTorch is
   available out-of-the-box, and support for many other model types,
   including models from scikit-learn, are in active development.
-  **Hierarchical Federated Learning for high scalability**. FEDn allows
   for massive horizontal scaling. This is achieved by a hierarchical
   architecture where multiple combiners divide up the work to
   coordinate client updates and aggregation. Recent benchmarks show
   high performance both for thousands of clients in a cross-device
   setting and for large model updates (1GB) in a cross-silo setting,
   see https://arxiv.org/abs/2103.00148.
-  **Built for real-world production scenarios**. The implementation is
   based on proven design patterns in distributed computing and
   incorporates enterprise security features. A key feature is that
   clients do not have to expose any ingress ports.
-  **UI to manage alliances, track training progress and follow client
   validations in real time**. FEDn has a UI that lets you efficiently
   track events, training progress, visualize model validations accross
   the federation, and monitor both client and server performance.

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
Keras example <examples/mnist-keras>`__.

Distributed deployment
======================

We provide instructions for a distributed reference deployment here:
`Distributed
deployment <https://scaleoutsystems.github.io/fedn/#/deployment>`__.

Where to go from here
=====================

-  `Explore additional examples <examples>`__
-  `Understand the
   architecture <https://scaleoutsystems.github.io/fedn/#/architecture>`__
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

-  `Documentation <https://scaleoutsystems.github.io/fedn/>`__ 
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

.. |FEDn logo| image:: https://github.com/scaleoutsystems/fedn/raw/master/docs/source/img/logos/Scaleout.png
   :width: 15%
.. |UU logo| image:: https://github.com/scaleoutsystems/fedn/raw/master/docs/source/img/logos/UU.png
   :width: 15%
.. |AI Sweden logo| image:: https://github.com/scaleoutsystems/fedn/raw/master/docs/source/img/logos/ai-sweden-logo.png
   :width: 15%
.. |Zenseact logo| image:: https://github.com/scaleoutsystems/fedn/raw/master/docs/source/img/logos/zenseact-logo.png
   :width: 15%
.. |Scania logo| image:: https://github.com/scaleoutsystems/fedn/raw/master/docs/source/img/logos/Scania.png
   :width: 15%