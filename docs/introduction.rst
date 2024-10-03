What is Federated Learning? 
===========================

Federated Learning is a novel approach to address challenges related to data privacy, security, 
and decentralized data distribution. In contrast to traditional machine learning setups where data is collected and stored centrally, 
Federated Learning allows for collaborative model training while keeping data local with the data owner or device. This is particularly advantageous in 
scenarios where data cannot be easily shared due to privacy regulations, network limitations, or ownership concerns.

At its core, Federated Learning orchestrates model training across distributed devices or servers, referred to as clients or participants. 
These participants could be diverse endpoints such as mobile devices, IoT gateways, or remote servers. Rather than transmitting raw data to a central location, 
each participant computes gradients locally based on its data. These gradients are then communicated to a server, often called the aggregator. 
The server aggregates and combines the gradients from multiple participants to update a global model. 
This iterative process allows the global model to improve without the need to share the raw data.


The FEDn framework 
--------------------

The goal with FEDn is to provide a federated learning framework that is secure, scalable and easy-to-use. Our ambition is that FEDn supports the full journey from early
testing/exploration, through pilot projects, to real-world depoyments and integration. We believe that that minimal code change should be needed to progress from early proof-of-concepts to production. This is reflected in our core design: 

-  **Minimal server-side complexity for the end-user**. Running a proper distributed FL deployment is hard. With FEDn Studio we seek to handle all server-side complexity and provide a UI, REST API and a Python interface to help users manage FL experiments and track metrics in real time.

-  **Secure by design.** FL clients do not need to open any ingress ports. Industry-standard communication protocols (gRPC) and token-based authentication and RBAC (Jason Web Tokens) provides flexible integration in a range of production environments.  

-  **ML-framework agnostic**. A black-box client-side architecture lets data scientists interface with their framework of choice. 

-  **Cloud native.** By following cloud native design principles, we ensure a wide range of deployment options including private cloud and on-premise infrastructure. 

-  **Scalability and resilience.** Multiple aggregation servers (combiners) can share the workload. FEDn seamlessly recover from failures in all critical components and manages intermittent client-connections. 

-  **Developer and DevOps friendly.** Extensive event logging and distributed tracing enables developers to monitor the sytem in real-time, simplifying troubleshooting and auditing. Extensions and integrations are facilitated by a flexible plug-in architecture.  

Features
--------

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

- Python client (`FEDn C++ client <https://github.com/scaleoutsystems/fedn>`__)
- C++ client (`FEDn C++ client <https://github.com/scaleoutsystems/fedn-cpp-client>`__)
- Android Kotlin client (`FEDn Kotlin client <https://github.com/scaleoutsystems/fedn-android-client>`__)

Support
--------

Community support in available in our `Discord
server <https://discord.gg/KMg4VwszAd>`__.

For professionals / Enteprise, we offer `Dedicated support <https://www.scaleoutsystems.com/start#pricing>`__.

.. meta::
    :description lang=en:
        In contrast to traditional machine learning setups where data is collected and stored centrally, Federated Learning allows for collaborative model training while keeping data local with the data owner or device.
    :keywords: Federated Learning, Machine Learning, What is federated machine learning, Federated Learning Framework, Federated Learning Platform
    :og:title: What is Federated Learning?
    :og:description: Federated Learning is a novel approach to address challenges related to data privacy, security, and decentralized data distribution.
    :og:image: https://fedn.scaleoutsystems.com/static/images/scaleout_black.png
    :og:url: https://fedn.scaleoutsystems.com/docs/introduction.html
    :og:type: website