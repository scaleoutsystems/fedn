.. _agg-label:

Aggregators
===========

Overview
---------
Aggregators handle combinations of model updates received by the combiner into a combiner-level global model. 
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


Using different aggregators
----------------------------

FEDn supports the following aggregation algorithms: 

- FedAvg (default)
- FedAdam 
- FedYogi
- FedAdaGrad

The implementation of the methods from the FedOpt family follows the implemenation in this paper: https://arxiv.org/pdf/2003.00295.pdf 


Training sessions can be configured to use a given aggregator method. For example, to use FedAdam:

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
                    "model_id": seed_model['model'],
                    "rounds": 10
                 }

   result_fedadam = client.start_session(**session_config)

.. note:: 

   The FedOpt methods use server-side momentum. FEDn resets the aggregator for each new session. This means that momentum terms 
   will also be reset, i.e. the history will be forgotten. When using FedAdam, FedYogi and FedAdaGrad, the user needs to strike a 
   balance between the number of rounds in the session from a convergence and utility perspective.  

Several additional parameters that guide general behavior of the aggregation flow can be configured: 

- Round timeout: The maximal time the combiner waits before processing the update queue.  
- Buffer size: The maximal allowed length of the queue before processing it.
- Whether to retain or delete model update files after they have been processed (default is to delete them)

See API documenation for 

Extending FEDn with new aggregators
-----------------------------------

A developer can extend FEDn with his/her own Aggregator(s) by implementing the interface specified in 
:py:mod:`fedn.network.combiner.aggregators.aggregatorbase.AggregatorBase`. The developer implements two following methods:  

- ``on_model_update`` (optional)
- ``combine_models``

**on_model_update**

The on_model_update has access to the complete model update including the metadata passed on  by the clients (as specified in the training entrypoint, 
see compute package). The base class implements a default callback that checks that all metadata assumed by the aggregation algorithms FedAvg and FedOpt 
is present in the metadata. However, the callback could also be used to implement custom preprocessing and additional checks including strategies 
to filter out updates that are suspected to be corrupted or malicious. 

**combine_models**

This method is responsible for processing the model update queue and in doing so produce an aggregated model. This is the main extension point where the numerical detail of the aggregation scheme is implemented. The best way to understand how to implement this methods is to study the already implemented algorithms: 

- :py:mod:`fedn.network.combiner.aggregators.fedavg`
- :py:mod:`fedn.network.combiner.aggregators.fedopt`

To add an aggregator plugin ``myaggregator``, the developer implements the interface and places a file called ‘myaggregator.py’ in the folder ‘fedn.network.combiner.aggregators’. 


