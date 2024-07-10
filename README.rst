|pic1| |pic2| |pic3|

.. |pic1| image:: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml/badge.svg
   :target: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml

.. |pic2| image:: https://badgen.net/badge/icon/discord?icon=discord&label
   :target: https://discord.gg/KMg4VwszAd

.. |pic3| image:: https://readthedocs.org/projects/fedn/badge/?version=latest&style=flat
   :target: https://fedn.readthedocs.io

FEDn
--------

FEDn empowers its users to create federated learning applications that seamlessly transition from proof-of-concepts to secure real-world distributed deployments. 

Core design principles:

-  **Secure by design.** FL clients do not need to open any ingress ports. Industry-standard communication protocols (gRPC) and token-based authentication and RBAC (JWT) provides flexible integration in a range of production environments.  

-  **Cloud native.**. Minimal code change should be required to go from development and testing to production. By following cloud native design principles, we ensure a wide range of deployment options including private cloud and on-premise infrastructure.

-  **Scalability and resilience.** Multiple aggregation servers (combiners) can share the workload. FEDn seamlessly recover from failures in all critical components and manages intermittent client-connections. 

-  **Data-scientist friendly**. A ML-framework agnostic design lets data scientists implement use-cases using their framework of choice. An intuitive UI and a Python API enables managment of complex FL experiments and tracking user-defined metrics in real time.

-  **Developer friendly.** Extensive event logging and distributed tracing enables developers to monitor both the system and experiments in real-time, simplifying troubleshooting and auditing. 


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


FEDn Studio - From development to FL in production: 

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

1. Sign up for a `Free FEDn Studio account <https://fedn.scaleoutsystems.com/signup>`__
2. Take the `Quickstart tutorial <https://fedn.readthedocs.io/en/stable/quickstart.html>`__

FEDn Studio (SaaS) is free for academic use and personal development / small-scale testing and exploration. For users and teams requiring
additional project resources, dedicated support or other hosting options, `explore our plans <https://www.scaleoutsystems.com/start#pricing>`__.  

Documentation
=============

More details about the architecture, deployment, and how to develop your own application and framework extensions are found in the documentation:

-  `Documentation <https://fedn.readthedocs.io>`__


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
