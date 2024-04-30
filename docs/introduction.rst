What is FEDn? 
=============

Federated Learning offers a novel approach to address challenges related to data privacy, security, 
and decentralized data distribution. In contrast to traditional machine learning setups where data is collected and stored centrally, 
Federated Learning allows for collaborative model training while keeping data local with the data owner or device. This is particularly advantageous in 
scenarios where data cannot be easily shared due to privacy regulations, network limitations, or ownership concerns.

At its core, Federated Learning orchestrates model training across distributed devices or servers, referred to as clients or participants. 
These participants could be diverse endpoints such as mobile devices, IoT gateways, or remote servers. Rather than transmitting raw data to a central location, 
each participant computes gradients locally based on its data. These gradients are then communicated to a server, often called the aggregator. 
The server aggregates and combines the gradients from multiple participants to update a global model. 
This iterative process allows the global model to improve without the need to share the raw data.

FEDn empowers users to create federated learning applications that seamlessly transition from local proofs-of-concept to secure distributed deployments. 
We develop the FEDn framework following these core design principles:

-  **Seamless transition from proof-of-concepts to real-world FL**. FEDn has been designed to make the journey from R&D to real-world deployments as smooth as possibe. Develop your federated learning use case in a pseudo-local environment, then deploy it to FEDn Studio (cloud or on-premise) for real-world scenarios. No code change is required to go from development and testing to production. 

-  **Designed for scalability and resilience.** FEDn enables model aggregation through multiple aggregation servers sharing the workload. A hierarchical architecture makes the framework well suited borh for cross-silo and cross-device use-cases. FEDn seamlessly recover from failures in all critical components, and manages intermittent client-connections, ensuring robust deployment in production environments.

-  **Secure by design.** FL clients do not need to open any ingress ports, facilitating distributed deployments across a wide variety of settings. Additionally, FEDn utilizes secure, industry-standard communication protocols and supports token-based authentication and RBAC for FL clients (JWT), providing flexible integration in production environments.   

-  **Developer and data scientist friendly.** Extensive event logging and distributed tracing enables developers to monitor experiments in real-time, simplifying troubleshooting and auditing. Machine learning metrics can be accessed via both a Python API and visualized in an intuitive UI that helps the data scientists analyze and communicate ML-model training progress.


Features
=========

Federated machine learning: 

- Support for any ML framework (e.g. PyTorch, Tensforflow/Keras and Scikit-learn)
- Extendable via a plug-in architecture (aggregators, load balancers, object storage backends, databases  etc.)
- Built-in federated algorithms (FedAvg, FedAdam, FedYogi, FedAdaGrad, etc.)
- CLI and Python API client for running FEDn networks and coordinating experiments. 
- Implement clients in any language (Python, C++, Kotlin etc.)
- No open ports needed client-side.


FEDn Studio - From development to FL in production: 

-  Leverage Scaleout's free managed service for development and testing in real-world scenarios (SaaS).      
-  Token-based authentication (JWT) and role-based access control (RBAC) for FL clients.  
-  REST API and UI. 
-  Data science dashboard for orchestrating experiments and visualizing results.
-  Admin dashboard for managing the FEDn network and users/clients.
-  View extensive logging and tracing information. 
-  Collaborate with other data-scientists on the project specification in a shared workspace. 
-  Cloud or on-premise deployment (cloud-native design, deploy to any Kubernetes cluster)

Support
=========

Community support in available in our `Discord
server <https://discord.gg/KMg4VwszAd>`__.

Options are available for `Enterprise support <https://www.scaleoutsystems.com/start#pricing>`__.