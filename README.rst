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

Leverage both a flexible local development environment and a managed deployment of the server-side (SaaS) to accelerate development of real-world federated learning applications. 

Design principles:

-  **Seamless transition from proof-of-concepts to real-world FL**. FEDn has been designed to make the journey from R&D to real-world deployments as smooth as possibe. Develop your federated learning use case in a pseudo-local environment, then deploy it to FEDn Studio (cloud or on-premise) for real-world scenarios. No code change is required to go from development and testing to production. 

-  **Designed for massive scalability and resilience.** FEDn enables the coordination of clients and model aggregation through multiple aggregation servers sharing the workload. This hierarchical design makes the framework well suited borh for cross-silo and cross-device use-cases. FEDn seamlessly recover from failures in all critical components, and manages intermittent client-connections, ensuring robust deployment in production environments.

-  **Secure by design.** FL clients do not need to open any ingress ports, facilitating distributed deployments across a wide variety of settings. Additionally, FEDn utilizes secure, industry-standard communication protocols and supports token-based authentication and RBAC for FL clients (JWT), providing flexible integration in diverse production environments.   

-  **Developer and data scientist friendly.** Extensive event logging and distributed tracing enables developers to monitor experiments in real-time, simplifying troubleshooting and auditing processes. Machine learning metrics can be accessed via both a Python API and visualized in an intuitive UI that helps the data scientists analyze and communicate ML-model training progress. 


Features
=========

Federated machine learning: 

- Support for any ML framework (e.g. PyTorch, Tensforflow/Keras and Scikit-learn)
- Extendable via a plug-in architecture (aggregators, load balancers, object storage backends, databases  etc.)
- Built-in federated algorithms (FedAvg, FedAdam, FedYogi, FedAdaGrad, etc.) 
- Implement clients in any language (Python, C++, Kotlin etc.)
- No open ports needed client-side.


FEDn Studio: From development to FL in production: 

-  Leverage Scaleout's free managed service for development and testing in real-world scenarios (SaaS).      
-  Token-based authentication (JWT) and role-based access control (RBAC) for FL clients.  
-  REST API and UI. 
-  Data science dashboard for orchestrating experiments and visualizing results.
-  Admin dashboard for managing the FEDn network and users/clients.
-  View extensive logging and tracing information. 
-  Collaborate with other data-scientists on the project specification in a shared workspace. 
-  Cloud or on-premise deployment (cloud-native design, deploy to any Kubernetes cluster)


Getting started
============================

The best way to get started is to take the quickstart tutorial: 

- `Quickstart <https://fedn.readthedocs.io/en/latest/quickstart.html>`__

Documentation
=============

More details about the architecture, deployment, and how to develop your own application and framework extensions (such as custom aggregators) are found in the documentation:

-  `Documentation <https://fedn.readthedocs.io>`__


Running your project in FEDn Studio (SaaS or on-premise)
========================================================

The FEDn Studio SaaS is free for development, testing and research (one project per user, backend compute resources sized for dev/test):   

- `Register for a free account in FEDn Studio <https://studio.scaleoutsystems.com/signup/>`__
- `Take the tutorial to deploy your project on FEDn Studio <https://guide.scaleoutsystems.com/#/docs>`__  

Scaleout can also support users to scale up experiments and demonstrators on Studio, by granting custom resource quotas. Additonally, charts are available for self-managed deployment on-premise or in your cloud VPC (all major cloud providers). Contact the Scaleout team for more information.


Support
=================

Community support in available in our `Discord
server <https://discord.gg/KMg4VwszAd>`__.

Options are available for `Enterprise support <https://www.scaleoutsystems.com/start#pricing>`__.

Making contributions
====================

All pull requests will be considered and are much appreciated. For
more details please refer to our `contribution
guidelines <https://github.com/scaleoutsystems/fedn/blob/develop/CONTRIBUTING.md>`__.

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

Use of FEDn Studio (SaaS) is subject to the `Terms of Use <https://www.scaleoutsystems.com/terms>`__.
