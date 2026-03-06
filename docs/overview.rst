.. _scaleout_edge_overview:

Scaleout Edge Overview
======================

Scaleout Edge is a platform for **distributed MLOps and DataOps**. It enables organizations to train, deploy, and govern machine learning models across decentralized infrastructure—from on-premise data centers to edge devices—without ever moving the raw training data.

Traditional machine learning pipelines are centralized: data is collected, moved to a central lake, and processed on a cluster. Scaleout Edge **inverts this workflow**. It allows you to **bring the model to the data**.

By managing a "Global Model" that travels to your devices, learns from local data, and sends back only mathematical updates, Scaleout Edge solves the fundamental challenges of data gravity, privacy, and network constraints.


The Scaleout Platform
---------------------

Scaleout Edge provides the orchestration, security, and aggregation layers needed to run distributed AI at scale. It is designed to manage the full lifecycle of a decentralized project through three core functions:

* **Orchestrate:** Coordinate thousands of devices to participate in training rounds automatically.
* **Aggregate:** Securely combine model updates into a global model using hierarchical aggregation.
* **Govern:** Track lineage, versioning, and security across the entire network.


What Can I Use Scaleout Edge For?
---------------------------------

Scaleout Edge addresses critical modern ML deployment challenges:

* **Data Sovereignty and Privacy**  
  Train models on sensitive data (healthcare records, financial transactions, proprietary IP) that strictly cannot leave the device or premise due to **GDPR**, **HIPAA**, or internal compliance. The raw data never crosses the network; only the model weights do.

* **Bandwidth-Efficient Operations**  
  In edge environments (factories, satellites, mobile fleets), uploading terabytes of raw video or sensor data is cost-prohibitive or technically impossible. Scaleout Edge processes data locally and transmits only small model updates, **reducing network load by orders of magnitude**.

* **Resilient, Continuous Learning**  
  Models degrade over time. Instead of manually collecting new datasets to retrain, Scaleout Edge enables a **continuous loop** where devices constantly refine the model based on fresh, real-world data they encounter.


Scaleout Architecture
---------------------

Scaleout Edge uses a unique **three-tier architecture** designed for scalability and resilience in unstable network conditions. Unlike simple client-server setups, Scaleout introduces an aggregation layer to handle the complexity of the edge.

The architecture consists of:

* **The Controller (The Brain)**  
  The Controller is the central management service. It manages the **Global Model**, coordinates training rounds, and handles authentication. It acts as the **control plane** for the entire network. You interact with the Controller via the Web UI, CLI, or API.

* **The Combiner (The Aggregator)**  
  The Combiner is the **scalability engine** of the platform. It sits between the Controller and the Clients. Combiners can be deployed in the cloud or on edge gateways (near-edge). Their job is to receive model updates from devices, **aggregate them**, and send a single update up the stack. This hierarchical approach allows the system to scale to thousands of clients without bottling-necking the central server.

* **The Client (The Worker)**  
  The Client acts as the interface between the Scaleout platform and your local data. It runs on the edge device (IoT device, server, laptop). The Client executes the training code locally, manages on-device data access, and communicates with the Combiner.


Key Concepts
------------

* **The Project:** A Project is the workspace for a specific machine learning objective. It defines the network of clients, the machine learning framework being used, and the configuration for how training should proceed.

* **The Compute Package:** To train a model, you upload a Compute Package. This is a code bundle (typically Python) that contains your model definition and training logic. Scaleout Edge distributes this package to selected clients automatically at the start of a session.

* **The Round:** Training happens in rounds. In a single round:
   1. The Controller instructs clients to train.
   2. Clients download the latest Global Model and the Compute Package.
   3. Clients train on their local data and upload a model update.
   4. Combiners aggregate these updates.
   5. A new Global Model is committed.

* **The Global Model:** The Global Model is the shared intelligence of the network. It is the result of aggregating updates from all participating clients. It serves as the "**master**" version that is versioned, tracked, and eventually deployed for inference.
