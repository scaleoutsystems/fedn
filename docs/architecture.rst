.. _architecture-label:

Architecture overview
=====================

This page provides an overview of the **Scaleout Edge architecture**. What 
follows is a conceptual description of the components that make up a Scaleout 
Edge network and how they interact during a federated training session.

A Scaleout Edge network consists of three tiers:

- **Tier 1: Clients**
- **Tier 2: Combiners**
- **Tier 3: Controller and supporting services**
 
.. image:: img/Scaleout_Edge_network.png
   :alt: Scaleout Edge network
   :width: 100%
   :align: center

Tier 1 — Clients
----------------

A **Client** (gRPC client) is a data node holding private data and connecting to
a Combiner (gRPC server) to receive training tasks and validation requests during
federated sessions.

Key characteristics:

- Clients communicate **outbound only** using RPC.  
  No inbound or publicly exposed ports are required.
- Upon connecting to the network, a client receives a **compute package** from the 
  Controller or uses one that is locally available for the client. This package 
  contains training and validation code to execute locally.
- The compute package is defined by entry points in the client code and can be
  customized to support various model types, frameworks, and even programming
  languages.

Python, C++ and Kotlin client implementations are provided out-of-the-box, but clients may be
implemented in any language to suit specific hardware or software environments.

Tier 2 — Combiners
------------------

A **Combiner** orchestrates and aggregates model updates coming from its
group of clients. It is responsible for the mid-level federated learning workflow.

Key responsibilities:

- Running a dedicated gRPC server for interacting with clients and the Controller.
- Executing the orchestration plan defined in the global **compute plan**
  provided by the Controller.
- Reducing client model updates into a single **combiner-level model**.

Because each Combiner operates independently, the total number of clients that
can be supported scales with the number of deployed Combiners. Combiners may be
placed in the cloud, on fog/edge nodes, or in any environment suited for running
the aggregation service.

Tier 3 — Controller and base services
-------------------------------------

Tier 3 contains several services, with the **Controller** being the central
component coordinating global training. The Controller has three primary roles:

1. **Global orchestration**  
   It defines the overall training strategy, distributes the compute plan, and
   specifies how combiner-level models should be combined into a global model.

2. **Global state management**  
   The Controller maintains the **model trail**—an immutable record of global
   model updates forming the training timeline.

3. **Discovery and connectivity**  
   It provides discovery services and mediates connections between clients and
   combiners. For this purpose, the Controller exposes a standard REST API used
   by RPC clients/servers and by user interfaces.

Additional Tier 3 services include:

- **Reducer**  
  Aggregates the combiner-level models into a single global model.

- **StateStore**  
  Stores the state of the network, training sessions, and metadata.

- **ModelRegistry**  
  Stores the final global model trail after a completed training session.

Notes on aggregation algorithms
-------------------------------

Scaleout Edge includes several **built-in aggregators** for common FL workflows
(see :ref:`agg-label`). For advanced scenarios, users may override the
Combiner-level behavior using **server functions** (:ref:`server-functions`),
allowing custom orchestration or aggregation logic.

Aggregation happens in two stages:  
1) each Combiner reduces client updates into a *combiner-level model*, and  
2) the Controller (Reducer) combines these into the final global model.

.. meta::
   :description lang=en:
      Architecture overview - An overview of the Scaleout Edge federated learning platform architecture.
   :keywords: Federated Learning, Architecture, Federated Learning Framework, Federated Learning Platform, FEDn, Scaleout Systems, Scaleout Edge
   
