.. _agg-label:

Aggregators
===========

Aggregators handle combinations of model updates received by the combiner into a combiner-level global model. 
During a training session, the combiners will instantiate an Aggregator and use it to process the incoming model updates from clients.

.. image:: img/aggregators.png
   :alt: Aggregator overview
   :width: 100%
   :align: center

The above figure illustrates the overall flow. When a client completes a model update, the model parameters are streamed to the combiner, and a model update message is sent. The model parameters are written to file on disk, and the model update message is passed to a callback function, on_model_update. The callback function validates the model update, and if successful, puts the update message on an aggregation queue. The model parameters are written to disk at a configurable storage location at the combiner. This is done to avoid exhausting RAM memory at the combiner. As multiple clients send updates, the aggregation queue builds up, and when a certain criteria is met, another method, combine_models, starts processing the queue, aggregating models according to the specifics of the scheme (FedAvg, FedAdam, etc). 

The user can configure several parameters that guide general behavior of the aggregation flow: 

- Round timeout: The maximal time the combiner waits before processing the update queue.  
- Buffer size: The maximal allowed length of the queue before processing it.
- Whether to retain or delete model update files after they have been processed (default is to delete them)



A developer can extend FEDn with his/her own Aggregator(s) by implementing the interface specified in 
:py:mod:`fedn.network.combiner.aggregators.aggregatorbase.AggregatorBase`. The developer implements two following methods:  

- ``on_model_update`` (optional)
- ``combine_models``

on_model_update
----------------

The on_model_update has access to the complete model update including the metadata passed on  by the clients (as specified in the training entrypoint, see compute package).  The base class implements a default callback that checks that all metadata assumed by the aggregation algorithms FedAvg and FedAdam is present in the metadata. However, the callback could also be used to implement custom preprocessing and additional checks including strategies to filter out updates that are suspected to be corrupted or malicious. 

combine_models
--------------

This method is responsible for processing the model update queue and in doing so produce an aggregated model. This is the main extension point where the numerical detail of the aggregation scheme is implemented. The best way to understand how to implement this methods is to study the already implemented algorithms: 

- :py:mod:`fedn.network.combiner.aggregators.fedavg`
- :py:mod:`fedn.network.combiner.aggregators.fedopt`

To add an aggregator plugin “myaggregator”, the developer implements the interface and places a file called ‘myaggregator.py’ in the folder ‘fedn.network.combiner.aggregators’. 


