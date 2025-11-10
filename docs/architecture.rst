.. _architecture-label:

Architecture overview
=====================

Constructing a federated model with Scaleout Edge amounts to a) specifying the details of the client-side training code and data integrations, and b) deploying the federated network. A Scaleout Edge network, as illustrated in the picture below, is made up of components into three different tiers: the *Controller* tier (3), one or more *Combiners* in second tier (2), and a number of *Clients* in tier (1). 
The combiners forms the backbone of the federated ML orchestration mechanism, while the Controller tier provides discovery services and controls to coordinate training over the federated network. 
By horizontally scaling the number of combiners, one can meet the needs of a growing number of clients.  
 
.. image:: img/Scaleout_Edge_network.png
   :alt: Scaleout Edge network
   :width: 100%
   :align: center




**The clients: tier 1**

A Client (gRPC client) is a data node, holding private data and connecting to a Combiner (gRPC server) to receive model update requests and model validation requests during training sessions. 
Importantly, clients uses remote procedure calls (RPC) to ask for model updates tasks, thus the clients not require any open ingress ports! A client receives the code (called package or compute package) to be executed from the *Controller* 
upon connecting to the network, and thus they only need to be configured prior to connection to read the local datasets during training and validation. The package is based on entry points in the client code, and can be customized to fit the needs of the user.
This allows for a high degree of flexibility in terms of what kind of training and validation tasks that can be performed on the client side. Such as different types of machine learning models and framework, and even programming languages.
A python3 client implementation is provided out of the box, and it is possible to write clients in a variety of languages to target different software and hardware requirements.  

**The combiners: tier 2**

A combiner is an actor whose main role is to orchestrate and aggregate model updates from a number of clients during a training session. 
When and how to trigger such orchestration are specified in the overall *compute plan* laid out by the *Controller*. 
Each combiner in the network runs an independent gRPC server, providing RPCs for interacting with the federated network it controls. 
Hence, the total number of clients that can be accommodated in a Scaleout Edge network is proportional to the number of active combiners in the Scaleout Edge network. 
Combiners can be deployed anywhere, e.g. in a cloud or on a fog node to provide aggregation services near the cloud edge. 

**The controller: tier 3**

Tier 3 does actually contain several components and services, but we tend to associate it with the *Controller* the most. The *Controller* fills three main roles in the Scaleout Edge network:

1. it lays out the overall, global training strategy and communicates that to the combiner network.
It also dictates the strategy to aggregate model updates from individual combiners into a single global model, 
2. it handles global state and maintains the *model trail* - an immutable trail of global model updates uniquely defining the federated ML training timeline, and  
3. it provides discovery services, mediating connections between clients and combiners. For this purpose, the *Controller* exposes a standard REST API both for RPC clients and servers, but also for user interfaces and other services.

Tier 3 also contain a *Reducer* component, which is responsible for aggregating combiner-level models into a single global model. Further, it contains a *StateStore* database, 
which is responsible for storing various states of the network and training sessions. The final global model trail from a traning session is stored in the *ModelRegistry* database. 


**Notes on aggregating algorithms**

Scaleout Edge is designed to allow customization of the FedML algorithm, following a specified pattern, or programming model. 
Model aggregation happens on two levels in the network. First, each Combiner can be configured with a custom orchestration and aggregation implementation, that reduces model updates from Clients into a single, *combiner level* model. 
Then, a configurable aggregation protocol on the *Controller* level is responsible for combining the combiner-level models into a global model. By varying the aggregation schemes on the two levels in the system, 
many different possible outcomes can be achieved. Good starting configurations are provided out-of-the-box to help the user get started. See :ref:`agg-label` and API reference for more details.


.. meta::
   :description lang=en:
      Architecture overview - An overview of the Scaleout Edge federated learning platform architecture.
   :keywords: Federated Learning, Architecture, Federated Learning Framework, Federated Learning Platform, FEDn, Scaleout Systems, Scaleout Edge
   
