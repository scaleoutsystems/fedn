Distributed deployment
======================

The actual deployment, sizing of nodes, and tuning of a FEDn network in production depends heavily on the use case (cross-silo, cross-device, etc), the size of model updates, on the available infrastructure, and on the strategy to provide end-to-end security. You can easily use the provided docker-compose templates to deploy FEDn network across different hosts in a live environment, but note that it might be necessary to modify them slightly depending on your target environment and host configurations.   

This example serves as reference deployment for setting up a fully distributed FEDn network consisting of one host serving the supporting services (Minio, MongoDB), one host serving the reducer, one host running two combiners, and one host running a variable number of clients. 

> Warning, there are additional security considerations when deploying a live FEDn network, outside of core FEDn functionality. Make sure to include these aspects in your deployment plans.

Prerequisite for the reference deployment
-----------------------------------------

Hosts
.....

This example assumes root access to 4 Ubuntu 20.04 Servers for running the FEDn network. We recommend at least 4 CPU, 8GB RAM flavors for the base services and the reducer, and 4 CPU, 16BG RAM for the combiner host. Client host sizing depends on the number of clients you plan to run. You need to be able to configure security groups/ingress settings for the service node, combiner, and reducer host.

Certificates
............

Certificates are needed for the reducer and combiner services. By default, FEDn will generate unsigned certificates for the reducer and combiner nodes using OpenSSL. 

> Certificates based on IP addresses are not supported due to incompatibilities with gRPC. 

1. Deploy supporting services  
-----------------------------

First, deploy Minio and Mongo services on one host (make sure to change the default passwords). Confirm that you can access MongoDB via the MongoExpress dashboard before proceeding with the reducer.  

2. Deploy the reducer
---------------------

Follow the steps for pseudo-distributed deployment, but now edit the settings-reducer.yaml file to provide the appropriate connection settings for MongoDB and Minio from Step 1. Also, copy 'config/extra-hosts-reducer.yaml.template' to 'config/extra-hosts-reducer.yaml' and edit it, adding a host:IP mapping for each combiner you plan to deploy. Then you can start the reducer: 

.. code-block:: bash

   sudo docker-compose -f config/reducer.yaml -f config/extra-hosts-reducer.yaml up 

3. Deploy combiners
-------------------

Edit 'config/settings-combiner.yaml' to provide a name for the combiner (used as a unique identifier for the combiner in the network), a hostname (which is used by reducer and clients to connect to combiner RPC), and the port (default is 12080, make sure to allow access to this port in your security group/firewall settings). Also, provide the IP and port for the reducer under the 'controller' tag. Then deploy the combiner: 

.. code-block:: bash

   sudo docker-compose -f config/combiner.yaml up 

Optional: Repeat the same steps for the second combiner node. Make sure to provide unique names for the two combiners. 

> Note that it is not currently possible to use the host's IP address as 'host'. This is due to gRPC not being able to handle certificates based on IP. 

4. Attach clients to the FEDn network
-------------------------------------

Once the FEDn network is deployed, you can attach clients to it in the same way as for the pseudo-distributed deployment. You need to provide clients with DNS information for all combiner nodes in the network. For example, to start 5 unique MNIST clients on a single host, copy  'config/extra-hosts-clients.template.yaml' to 'test/mnist-keras/extra-hosts.yaml' and edit it to provide host:IP mappings for the combiners in the network. Then, from 'test/mnist-keras':

.. code-block:: bash

   sudo docker-compose -f docker-compose.yaml -f config/extra-hosts-client.yaml up --scale client=5 
