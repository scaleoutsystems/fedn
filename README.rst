
.. image:: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml/badge.svg
   :target: https://github.com/scaleoutsystems/fedn/actions/workflows/integration-tests.yaml

.. image:: https://badgen.net/badge/icon/discord?icon=discord&label
   :target: https://discord.gg/KMg4VwszAd

.. image:: https://readthedocs.org/projects/fedn/badge/?version=latest&style=flat
   :target: https://fedn.readthedocs.io

FEDn
--------

FEDn empowers developers, researchers, and data scientists to create federated learning applications that seamlessly transition from local proofs-of-concept to real-world distributed deployments. Develop your federated learning use case in a pseudo-local environment, and deploy it to FEDn Studio for real-world Federated Learning without any need for code changes.

Core Features
=============

-  **Scalable and resilient.** FEDn facilitates the coordination of clients and model aggregation through multiple aggregation servers sharing the workload. This design makes the framework highly scalable, accommodating large numbers of clients. The system is engineered to seamlessly recover from failures, ensuring robust deployment in production environments. Furthermore, FEDn adeptly manages asynchronous federated learning scenarios, accommodating clients that may connect or drop out during training.

-  **Security**. FL clients do not need to open any ingress ports, facilitating real-world deployments across a wide variety of settings. Additionally, FEDn utilizes secure, industry-standard communication protocols and supports token-based authentication for FL clients, enhancing security and ease of integration in diverse environments.   

-  **Track events and training progress in real-time**. Extensive event logging and distributed tracing enable developers to monitor experiments in real-time, simplifying troubleshooting and auditing processes. Machine learning validation metrics from clients can be accessed via the API, allowing for flexible analysis of federated experiments. 

-  **ML-framework agnostic**. FEDn is compatible with all major ML frameworks. Examples for Keras, PyTorch and scikit-learn are
   available out-of-the-box.

From development to real-world FL: 

-  Develop a FEDn project in a local development environment, and then deploy it to FEDn Studio
-  The FEDn server-side as a managed, production-grade service on Kubernetes. 
-  Token-based authentication for FL clients  
-  Role-based access control (RBAC)
-  REST API 
-  Dashboard for orchestrating runs, visualizing and downloading results
-  Admin dashboard for managing and scaling the FEDn network 
-  Collaborate with other data-scientists in a shared workspace. 
-  Cloud or on-premise deployment 


Getting started with FEDn
===============

The best way to get started is to take the quickstart tutorial: 

- `Quickstart <https://fedn.readthedocs.io/en/latest/quickstart.html>`__

Documentation
=============

More details about the architecture, deployment, and how to develop your own application and framework extensions (such as custom aggregators) are found in the documentation:

-  `Documentation <https://fedn.readthedocs.io>`__


Deploying a project to FEDn Studio
===============

Studio offers a production-grade deployment of the FEDn server-side infrastructure on Kubernetes. With Studio, you can also manage token-based authentication for clients and collaborate with other users in joint project workspaces. In addition to a REST API, Studio features intuitive dashboards that allows you to orchestrate FL experiments and visualize and manage global models, event logs and metrics. These features enhance your ability to monitor and analyze federated learning projects. Studio is available as-a service hosted by Scaleout and one project is provided for free for testing and research. 

- `Register for a project in Studio <https://studio.scaleoutsystems.com/signup/>`__
- `Deploy you project to FEDn Studio <https://guide.scaleoutsystems.com/#/docs>`__  

Options and charts are also available for self-managed deployment of FEDn Studio, reach out to the Scaleout team for more information. 


Support
=================

Community support in available in our `Discord
server <https://discord.gg/KMg4VwszAd>`__.

Options are also available for `Enterprise support <https://www.scaleoutsystems.com/start#pricing>`__.

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
