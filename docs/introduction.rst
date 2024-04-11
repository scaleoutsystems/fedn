What is FEDn? 
==================================

Federated Learning offers a novel approach to address challenges related to data privacy, security, 
and decentralized data distribution. In contrast to traditional machine learning setups where data is collected and stored centrally, 
Federated Learning allows for collaborative model training while keeping data local with the data owner or device. This is particularly advantageous in 
scenarios where data cannot be easily shared due to privacy regulations, network limitations, or ownership concerns.

At its core, Federated Learning orchestrates model training across distributed devices or servers, referred to as clients or participants. 
These participants could be diverse endpoints such as mobile devices, IoT gadgets, or remote servers. Rather than transmitting raw data to a central location, 
each participant computes gradients locally based on its data. These gradients are then communicated to a server, often called the aggregator. 
The server aggregates and combines the gradients from multiple participants to update a global model. 
This iterative process allows the global model to improve without the need to share the raw data.

FEDn: the SDK for scalable federated learning
.............................................

FEDn serves as a System Development Kit (SDK) enabling scalable federated learning. 
It is used to implement the core server side logic (including model aggregation) and the client side integrations. 
Developers and ML engineers can use FEDn to build custom federated learning systems and bespoke deployments.


One of the standout features of FEDn is its ability to deploy and scale the server-side in geographically distributed setups,
adapting to varying project needs and geographical considerations.


Scalable and Resilient
......................

FEDn exhibits scalability and resilience, thanks to its tiered architecture. Multiple aggregation servers, in FEDn called combiners, 
form a network to divide the workload of coordinating clients and aggregating models. 
This architecture allows for high performance in various settings, from thousands of clients in a cross-device environment to 
large model updates in a cross-silo scenario. Importantly, FEDn has built-in recovery capabilities for all critical components, enhancing system reliability.

ML-Framework Agnostic
.....................

With FEDn, model updates are treated as black-box computations, meaning it can support any ML model type or framework. 
This flexibility allows for out-of-the-box support for popular frameworks like Keras and PyTorch, making it a versatile tool for any machine learning project.

Security
.........

A key security feature of FEDn is its client protection capabilities - clients do not need to expose any ingress ports, 
thus reducing potential security vulnerabilities.

Event Tracking and Training progress
....................................

To ensure transparency and control over the training process, as well as to provide means to troubleshoot distributed deployments, 
FEDn logs events and does real-time tracking of training progress. A flexible API lets the user define validation strategies locally on clients. 
Data is logged as JSON to MongoDB, enabling users to create custom dashboards and visualizations easily.

REST-API and Python API Client
...............

FEDn comes with an REST-API, a CLI and a Python API Client for programmatic interaction with a FEDn network. This allows for flexible automation of experiments, for integration with 
other systems, and for easy integration with external dashboards and visualization tools.