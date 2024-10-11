What is Federated Learning? 
===========================

Federated learning (FL) is a decentralized approach to machine learning. It tackles the issues of centralized machine learning by allowing models to train on distributed data sources. Instead of moving the data, FL moves the computation to where the data is. The result is then combined into a globally-informed model, all this while preserving data privacy and security.

Traditional machine learning
-----------------------------

Traditional machine learning utilizes a centralized approach. This  involves collecting data from various sources into one, centralized repository. This often being a cloud environment or a dedicated data center.  Then algorithms are used to train models on the dataset. The resulting models can be deployed to make decisions based on new incoming data.

This often works well, but the centralized approach to machine learning is facing challenges. There has been a  rapid increase of connected devices, sensors and distributed data sources. This in turn has led to an exponential increase in the volume and complexity of data being generated. At the same time, privacy, security and compliance concerns have made it harder to move and combine data from different sources.

The data needed to train machine learning models is often distributed. It can exist across several organizations, devices, or clients. This makes centralization challenging due to privacy risks and high transfer costs.

.. image:: img/machine_learning_centralized_decentralized.svg

How federated learning works
-----------------------------

In federated learning, AI models are trained across multiple devices or servers (called client nodes) without needing to move the data off those devices. Here’s a simplified breakdown of how it works:

1. **Starting the global model -** The process begins with a global model on a central server. This could be any type of machine learning model, like a neural network or decision tree.
2. **Sending the model to clients -** The server sends the global model’s parameters to a group of selected client nodes. Each client uses its own local dataset, which stays securely on the device.
3. **Local training -** Each client trains the model using its local data, adjusting the model’s parameters based on what it learns from the data. This training process is repeated for several rounds, rather than continuing until full accuracy is achieved.
4. **Combining the updates -** The updated models from each client are sent back to the central server, where they are combined. A common approach is called Federated Averaging, where the server takes a weighted average of the updates from each client.

At last, the improved global model is sent back to the clients for further training. This cycle continues until the model reaches a satisfactory level of accuracy.

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
        Federated learning is a decentralized approach that tackles the issues of centralized machine learning by allowing models to be trained on data distributed across various locations without moving the data.
    :keywords: Federated Learning, Machine Learning, What is federated machine learning, Federated Learning Framework, Federated Learning Platform
    :og:title: What is Federated Learning?
    :og:description: Federated learning is a decentralized approach that tackles the issues of centralized machine learning by allowing models to be trained on data distributed across various locations without moving the data.
    :og:image: https://fedn.scaleoutsystems.com/static/images/scaleout_black.png
    :og:url: https://fedn.scaleoutsystems.com/docs/introduction.html
    :og:type: website