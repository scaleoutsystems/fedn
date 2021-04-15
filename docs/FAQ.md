# Frequently asked questions

### Q: How do I remove/replace the compute package?

We do not provide an out-of-the box way to clear the compute package for a model that has been intitialized. 
This is a security constraint enforced to not allow for arbitrary code package replacement in an already configured federation. 
Once the federated model has been initilized and seeded it should be seeen as immutable. However, during development of a new model
it will be necessary to reinitialize. Then you can follow this procedure: 

  1. Clear the database. Navigate to http://localhost:8081 and delete the entire "fedn-test-network" collection. 
  2. Restart the reducer, combiner and reattach the clients. 

There are also additional ways to enable interative development by bypassing the need to use/upload a compute package.  


 
