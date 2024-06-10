|pic1| |pic2| |pic3|

.. |pic1| image:: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml/badge.svg
   :target: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml

.. |pic2| image:: https://badgen.net/badge/icon/discord?icon=discord&label
   :target: https://discord.gg/KMg4VwszAd

.. |pic3| image:: https://readthedocs.org/projects/fedn/badge/?version=latest&style=flat
   :target: https://fedn.readthedocs.io

FEDn
--------

FEDn empowers its users to create federated learning applications that seamlessly transition from local proofs-of-concept to secure distributed deployments. 

Leverage a flexible pseudo-local sandbox to rapidly transition your existing ML project to a federated setting. Test and scale in real-world scenarios using FEDn Studio - a fully managed, secure deployment of all server-side components (SaaS). 

We develop the FEDn framework following these core design principles:

-  **Seamless transition from proof-of-concepts to real-world FL**. FEDn has been designed to make the journey from R&D to real-world deployments as smooth as possibe. Develop your federated learning use case in a pseudo-local environment, then deploy it to FEDn Studio (cloud or on-premise) for real-world scenarios. No code change is required to go from development and testing to production. 

-  **Designed for scalability and resilience.** FEDn enables model aggregation through multiple aggregation servers sharing the workload. A hierarchical architecture makes the framework well suited borh for cross-silo and cross-device use-cases. FEDn seamlessly recover from failures in all critical components, and manages intermittent client-connections, ensuring robust deployment in production environments.

-  **Secure by design.** FL clients do not need to open any ingress ports, facilitating distributed deployments across a wide variety of settings. Additionally, FEDn utilizes secure, industry-standard communication protocols and supports token-based authentication and RBAC for FL clients (JWT), providing flexible integration in production environments.   

-  **Developer and data scientist friendly.** Extensive event logging and distributed tracing enables developers to monitor experiments in real-time, simplifying troubleshooting and auditing. Machine learning metrics can be accessed via both a Python API and visualized in an intuitive UI that helps the data scientists analyze and communicate ML-model training progress.


Features
=========

Core FL framework (this repository): 

- Tiered federated learning architecture enabling massive scalability and resilience. 
- Support for any ML framework (examples for PyTorch, Tensforflow/Keras and Scikit-learn)
- Extendable via a plug-in architecture (aggregators, load balancers, object storage backends, databases  etc.)
- Built-in federated algorithms (FedAvg, FedAdam, FedYogi, FedAdaGrad, etc.)
- CLI and Python API.
- Implement clients in any language (Python, C++, Kotlin etc.)
- No open ports needed client-side.
- Flexible deployment of server-side components using Docker / docker compose.   


FEDn Studio - From development to FL in production: 

-  Secure deployment of server-side / control-plane on Kubernetes.
-  UI with dashboards for orchestrating experiments and visualizing results
-  Team features - collaborate with other users in shared project workspaces. 
-  Features for the trusted-third party: Manage access to the FL network, FL clients and training progress.
-  REST API for handling experiments/jobs. 
-  View and export logging and tracing information. 
-  Public cloud, dedicated cloud and on-premise deployment options.

Available clients:

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
