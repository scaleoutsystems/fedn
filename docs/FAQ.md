# Frequently asked questions

### Q: How do I remove/replace the compute package?

We do not provide an out-of-the box way to clear the compute package for a model that has been intitialized. 
This is a security constraint enforced to not allow for arbitrary code package replacement in an already configured federation. 
Once the federated model has been initilized and seeded it should be seeen as immutable. However, during development of a new model
it will be necessary to reinitialize. Then you can follow this procedure: 

  1. Clear the database. Navigate to http://localhost:8081 and delete the entire "fedn-test-network" collection. 
  2. Restart the reducer, combiner and reattach the clients. 

There are also additional ways to enable interative development by bypassing the need to use/upload a compute package.  

### Q: How can other aggregation algorithms can be defined?

Currently it is possible for the advanced user to implement/extend the aggregation routine at the combiner level by modifying or replacing the following class:  
https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/algo/fedavg.py 

In a future release, we plan to extend the aggregation API to make this process more standardized and configurable at runtime. 

### Q: What is needed to include other ML frameworks in FEDn like sklearn, xgboost, etc.?

FEDn has a black-box model for the model update by clients. What is assumed in the framework are entrypoints/binaries "train.py" and "validate.py" that operate in a SISO fashion: 

    $ ./train.py model_update_in model_update_out

and
    $ ./validate.py model_update_in validation.json 

These are defined in the client, and are part of the compute package. To support a yet unseen framework or model type, the user needs to implement a helper class that handles serialization and deserialization of the input and output, as well as the aggregation that happens in the combiner. See the following files for examples: 

https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/utils/helpers.py
https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/utils/kerashelper.py 
https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/utils/pytorchhelper.py

Naturally, the model needs to be of a type where it makes sense to perform aggregation of model parameters in FedAvg or variants thereof. 

### Q: How can I configure the round participation policy: 

In the main control implementation https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/clients/reducer/control.py you can modify or replace the instance variables "self.client_allocation_policy" with your own implementation to modify the client assignment behavior (see also the methods "client_allocation_policy_first_available" and "client_allocation_policy_least_packed"). In a future release, we plan to make this runtime configurable. 

### Q: How can I configure the round validity policy: 

In the main control implementation https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/clients/reducer/control.py you can modify or replace the wiwmethod "check_round_validity_policy". As we expand with more implementations of this policy, we plan to make it runtime configurable. 

### Q: Do you plan to support additional privacy-preserving mechanisms and frameworks? 

Yes - we have focused on scalability in a tiered aggregation model, and Enterprise security features in early development. But we plan to add addtional security mechanisms, such as differential privacy and multiparty computation (between client-combiner or for the reduce protocol). Please let us know what you need in your projects to help us prioritize development.   

### Q: In the current implementaion, the Reducer is a serialization point, do you plan to provide additional 

Yes. In a near future we will generalize that part into a _Reduce protocol_ executing over the combiner network. This will enable many different improved schemes such as ring allreduce, hierararchical/tree-based reduce, and even fully decentralized protocols (e.g. gossip protocols). 

