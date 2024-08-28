|pic1| |pic2| |pic3|

.. |pic1| image:: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml/badge.svg
   :target: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml

.. |pic2| image:: https://badgen.net/badge/icon/discord?icon=discord&label
   :target: https://discord.gg/KMg4VwszAd

.. |pic3| image:: https://readthedocs.org/projects/fedn/badge/?version=latest&style=flat
   :target: https://fedn.readthedocs.io

FEDn: An enterprise-ready federated learning framework 
-------------------------------------------------------

Our goal is to provide a federated learning framework that is both secure, scalable and easy-to-use. We believe that that minimal code change should be needed to progress from early proof-of-concepts to production. This is reflected in our core design: 

-  **Minimal server-side complexity for the end-user**. Running a proper distributed FL deployment is hard. With FEDn Studio we seek to handle all server-side complexity and provide a UI, REST API and a Python interface to help users manage FL experiments and track metrics in real time.

-  **Secure by design.** FL clients do not need to open any ingress ports. Industry-standard communication protocols (gRPC) and token-based authentication and RBAC (Jason Web Tokens) provides flexible integration in a range of production environments.  

-  **ML-framework agnostic**. A black-box client-side architecture lets data scientists interface with their framework of choice. 

-  **Cloud native.** By following cloud native design principles, we ensure a wide range of deployment options including private cloud and on-premise infrastructure. 

-  **Scalability and resilience.** Multiple aggregation servers (combiners) can share the workload. FEDn seamlessly recover from failures in all critical components and manages intermittent client-connections. 

-  **Developer and DevOps friendly.** Extensive event logging and distributed tracing enables developers to monitor the sytem in real-time, simplifying troubleshooting and auditing. Extensions and integrations are facilitated by a flexible plug-in architecture.  

FEDn is free forever for academic and personal use / small projects. Sign up for a `FEDn Studio account <https://fedn.scaleoutsystems.com/signup>`__  and take the `Quickstart tutorial <https://fedn.readthedocs.io/en/stable/quickstart.html>`__ to get started with FEDn. 

Features
=========

Federated learning: 

- Tiered federated learning architecture enabling massive scalability and resilience. 
- Support for any ML framework (examples for PyTorch, Tensforflow/Keras and Scikit-learn)
- Extendable via a plug-in architecture (aggregators, load balancers, object storage backends, databases  etc.)
- Built-in federated algorithms (FedAvg, FedAdam, FedYogi, FedAdaGrad, etc.)
- UI, CLI and Python API.
- Implement clients in any language (Python, C++, Kotlin etc.)
- No open ports needed client-side.


From development to FL in production: 

-  Secure deployment of server-side / control-plane on Kubernetes.
-  UI with dashboards for orchestrating FL experiments and for visualizing results
-  Team features - collaborate with other users in shared project workspaces. 
-  Features for the trusted-third party: Manage access to the FL network, FL clients and training progress.
-  REST API for handling experiments/jobs. 
-  View and export logging and tracing information. 
-  Public cloud, dedicated cloud and on-premise deployment options.

Available client APIs:

- Python client (this repository)
- C++ client (`FEDn C++ client <https://github.com/scaleoutsystems/fedn-cpp-client>`__)
- Android Kotlin client (`FEDn Kotlin client <https://github.com/scaleoutsystems/fedn-android-client>`__)


Getting started
============================

Get started with FEDn in two steps:  

1. Register for a `FEDn Studio account <https://fedn.scaleoutsystems.com/signup>`__
2. Take the `Quickstart tutorial <https://fedn.readthedocs.io/en/stable/quickstart.html>`__

Use of our multi-tenant, managed deployment of FEDn Studio (SaaS) is free forever for academic research and personal development/testing purposes.
For users and teams requiring additional resources, more storage and cpu, dedicated support, and other hosting options (private cloud, on-premise), `explore our plans <https://www.scaleoutsystems.com/start#pricing>`__.  

Documentation
=============

More details about the architecture, deployment, and how to develop your own application and framework extensions are found in the documentation:

-  `Documentation <https://fedn.readthedocs.io>`__

FEDn Project Examples
=====================

Our example projects demonstrate different use case scenarios of FEDn 
and its integration with popular machine learning frameworks like PyTorch and TensorFlow.

- `FEDn + PyTorch <https://github.com/scaleoutsystems/fedn/tree/master/examples/mnist-pytorch>`__
- `FEDn + Tensforflow/Keras <https://github.com/scaleoutsystems/fedn/tree/master/examples/mnist-keras>`__
- `FEDn + MONAI <https://github.com/scaleoutsystems/fedn/tree/master/examples/monai-2D-mednist>`__
- `FEDn + Hugging Face <https://github.com/scaleoutsystems/fedn/tree/master/examples/huggingface>`__
- `FEDn + Flower <https://github.com/scaleoutsystems/fedn/tree/master/examples/flower-client>`__
- `FEDN + Self-supervised learning <https://github.com/scaleoutsystems/fedn/tree/master/examples/FedSimSiam>`__

FEDn Studio Deployment options
==============================

Several hosting options are available to suit different project settings.

-  `Public cloud (multi-tenant) <https://fedn.scaleoutsystems.com>`__: Managed multi-tenant deployment in public cloud. 
-   Dedicated cloud (single-tenant): Managed, dedicated deployment in a cloud region of your choice (AWS, GCP, Azure, managed Kubernetes) 
-   Self-managed: Set up a self-managed deployment in your VPC or on-premise Kubernets cluster using Helm Chart and container images provided by Scaleout. 

Contact the Scaleout team for information.

Support
=================

Community support is available in our `Discord
server <https://discord.gg/KMg4VwszAd>`__.

Options are available for `Dedicated/custom support <https://www.scaleoutsystems.com/start#pricing>`__.

Making contributions
====================

All pull requests will be considered and are much appreciated. For
more details please refer to our `contribution
guidelines <https://github.com/scaleoutsystems/fedn/blob/master/CONTRIBUTING.md>`__.

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


License
=======

FEDn is licensed under Apache-2.0 (see `LICENSE <LICENSE>`__ file for
full information).

Use of FEDn Studio is subject to the `Terms of Use <https://www.scaleoutsystems.com/terms>`__.
