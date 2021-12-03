Frequently asked questions
==========================

Q: How do I remove/replace the compute package?
-----------------------------------------------

We do not provide an out-of-the box way to clear the compute package for a model that has been intitialized. 
This is a security constraint enforced to not allow for arbitrary code package replacement in an already configured federation. 
Once the federated model has been initilized and seeded it should be seeen as immutable. However, during development of a new model
it will be necessary to reinitialize. Then you can follow this procedure: 

  1. Kill the reducer, all combiners and all clients. 
  2. Clear the database: Navigate to http://localhost:8081 and delete the entire "fedn-test-network" collection. 
  3. Start the reducer, combiner and reattach the clients. 

There are also additional ways to enable interative development by bypassing the need to use/upload a compute package.  

Q: Can I skip fetching the remote package and instead use a local folder when developing the compute package
------------------------------------------------------------------------------------------------------------

Yes, to facilitate interactive development of the compute package you can start a client that uses a local folder 'client' in your current working directory by: 

.. code-block:: bash

    fedn run client --remote=False -in client.yaml 


Note that in production federations this options should in most cases be disallowed. 

Q: How can other aggregation algorithms can be defined?
-------------------------------------------------------

Yes. An advanced user can implement/extend the aggregation routine at the combiner level by modifying or replacing the following class:  
https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/aggregator/fedavg.py 

There is an abstract base class, 'AggregatorBase' in 'aggregator.py', for your reference. 

In a future release, we plan to extend the aggregation API to make this process more standardized and configurable at runtime. 

Q: What is needed to include other ML frameworks in FEDn like sklearn, xgboost, etc.?
-------------------------------------------------------------------------------------

FEDn has a black-box model for the model update by clients. What is assumed in the framework are entrypoints/binaries "train.py" and "validate.py" that operate in a SISO fashion: 

.. code-block:: bash

    $ ./train.py model_update_in model_update_out

and

.. code-block:: bash

    $ ./validate.py model_update_in validation.json 

These are defined in the client, and are part of the compute package. To support a yet unseen framework or model type, the user needs to implement a helper class that handles serialization and deserialization of the input and output, as well as the aggregation that happens in the combiner. See the following files for examples: 

https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/utils/helpers.py
https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/utils/kerashelper.py 
https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/utils/pytorchhelper.py

Naturally, the model needs to be of a type where it makes sense to perform aggregation of model parameters in FedAvg or variants thereof. 

Q: How can I configure the round participation policy: 
------------------------------------------------------

In the main control implementation https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/clients/reducer/control.py you can modify or replace the instance variables "self.client_allocation_policy" with your own implementation to modify the client assignment behavior (see also the methods "client_allocation_policy_first_available" and "client_allocation_policy_least_packed"). In a future release, we plan to make this runtime configurable. 

Q: How can I configure the round validity policy: 
-------------------------------------------------

In the main control implementation https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/clients/reducer/control.py you can modify or replace the wiwmethod "check_round_validity_policy". As we expand with more implementations of this policy, we plan to make it runtime configurable. 

Q: Can I start a client listening only to training requests or only on validation requests?:
-------------------------------------------------

Yes! From FEDn 0.3.0 there is an option to toggle which message streams a client subscibes to. For example, to start a pure validation client: 

.. code-block:: bash

    fedn run client --trainer=False -in client.yaml 


Q: Do you plan to support additional privacy-preserving mechanisms and frameworks? 
----------------------------------------------------------------------------------

Yes. We plan to add addtional security mechanisms, such as differential privacy and multiparty computation (between client-combiner or for the reduce protocol). Please let us know what you need in your projects to help us prioritize development.   


