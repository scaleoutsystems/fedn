.. _agg-label:

Aggregators
===========

Overview
---------
Aggregators are responsible for combining client model updates into a combiner-level global model. 
During a training session, the combiners will instantiate an Aggregator and use it to process the incoming model updates from clients.

.. image:: img/aggregators.png
   :alt: Aggregator overview
   :width: 100%
   :align: center

The figure above illustrates the overall workflow. When a client completes a model update, the model parameters are streamed to the combiner, 
and a model update message is sent. The parameters are saved to a file on disk, and the update message is passed to a callback function, ``on_model_update``. 
This function validates the model update and, if successful, places the update message in an aggregation queue. 
The model parameters are saved to disk at a configurable storage location within the combiner to prevent exhausting RAM. 
As multiple clients submit updates, the aggregation queue accumulates. Once specific criteria are met, another method, ``combine_models``, 
begins processing the queue, aggregating models according to the specifics of the scheme (e.g., FedAvg, FedAdam).


Using built-in Aggregators
--------------------------

FEDn supports the following aggregation algorithms: 

- FedAvg (default)
- FedAdam (FedOpt)
- FedYogi (FedOpt)
- FedAdaGrad (FedOpt)

The implementation of the methods from the FedOpt family follows the implemenation in this paper: https://arxiv.org/pdf/2003.00295.pdf 

Training sessions can be configured to use a given aggregator. For example, to use FedAdam:

.. code:: python

   session_config = {
                    "helper": "numpyhelper",
                    "id": "experiment_fedadam",
                    "aggregator": "fedopt",
                    "aggregator_kwargs": {
                        "serveropt": "adam",
                        "learning_rate": 1e-2,
                        "beta1": 0.9,
                        "beta2": 0.99,
                        "tau": 1e-4
                        },
                    "rounds": 10
                 }

   result_fedadam = client.start_session(**session_config)

.. note:: 

   The FedOpt family of methods use server-side momentum. FEDn resets the aggregator for each new session. 
   This means that the history will will also be reset, i.e. the momentum terms will be forgotten. 
   When using FedAdam, FedYogi and FedAdaGrad, the user needs to strike a 
   balance between the number of rounds in the session from a convergence and utility perspective.  

.. note:: 

   The parameter ``aggregator_kwargs`` are hyperparameters for the FedOpt family aggregators. The
   data types for these parameters (str, float) are validated by the aggregator and an error 
   will be issued if passing parameter values of incompatible type. All hyperparameters are 
   given above for completeness. It is primarily the ``learning_rate`` that will require tuning.

Several additional parameters that guide general behavior of the aggregation flow can be configured: 

- Round timeout: The maximal time the combiner waits before processing the update queue.  
- Buffer size: The maximal allowed length of the queue before processing it.
- Whether to retain or delete model update files after they have been processed (default is to delete them)


Extending FEDn with new Aggregators
-----------------------------------

A developer can extend FEDn with his/her own Aggregator(s) by implementing the interface specified in 
:py:mod:`fedn.network.combiner.aggregators.aggregatorbase.AggregatorBase`. This involes implementing the two methods:  

- ``on_model_update`` (perform model update validation before update is placed on queue, optional)
- ``combine_models``  (process the queue and aggregate updates)

**on_model_update**

The ``on_model_update`` callback recieves the model update messages from clients (including all metadata) and can be used to perform validation and 
potential transformation of the model update before it is placed on the aggregation queue (see image above). 
The base class implements a default callback that checks that all metadata assumed by the aggregation algorithms FedAvg and FedOpt is available. The callback could also be used to implement custom pre-processing and additional checks including strategies 
to filter out updates that are suspected to be corrupted or malicious. 

**combine_models**

When a certain criteria is met, e.g. if all clients have sent updates, or the round has times out, the ``combine_model_update`` method 
processes the model update queue, producing an aggregated model. This is the main extension point where the
numerical details of the aggregation scheme is implemented. The best way to understand how to implement this method is to study the built-in aggregation algorithms: 

- :py:mod:`fedn.network.combiner.aggregators.fedavg` (weighted average of parameters)
- :py:mod:`fedn.network.combiner.aggregators.fedopt` (compute pseudo-gradients and apply a server-side optmizer)

To add an aggregator plugin ``myaggregator``, the developer implements the interface and places a file called ‘myaggregator.py’ in the folder ‘fedn.network.combiner.aggregators’.
This extension can then simply be called as such: 

.. code:: python

   session_config = {
                    "helper": "numpyhelper",
                    "id": "experiment_myaggregator",
                    "aggregator": "myaggregator",
                    "rounds": 10
                 }

   result_myaggregator = client.start_session(**session_config)


.. meta::
   :description lang=en:
      Aggregators are responsible for combining client model updates into a combiner-level global model. During a training session, the combiners will instantiate an Aggregator and use it to process the incoming model updates from clients.
   :keywords: Federated Learning, Aggregators, Federated Learning Framework, Federated Learning Platform, FEDn, Scaleout Systems
