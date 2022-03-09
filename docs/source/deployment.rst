Distributed deployment
======================

FEDn is designed for flexibility at the deployment of the FEDn network. By deploying multiple combiners (aggregation servers) at separate hosts/locations, 
a user can run highly decentralized networks. However, in the degenerate case of a single active combiner, we obtain a simple central client-server architecture. 
The optimal choice of the network topology including the sizing of nodes in the network depends on many factors including the need for redundancy, 
the targeted number of clients, the size of the models trained, the need for geographical proximity of clients to combiners, and security considerations. 
Since it is easy to scale the network dynamically by attaching additional combiners, we recommend that you start simple and expand the network as needed.     

This example serves as reference deployment for setting up a fully distributed FEDn network consisting of:
   -  one host serving the supporting services (MinIOs, MongoDB)
   -  one host serving the controller / reducer 
   -  one host running a combiner 

We will in this example use the provided docker-compose templates to deploy the components across three different hosts / VMs. 
Note that it might be necessary to modify them slightly depending on your target environment and host configurations. 


Prerequisite 
-------------

Hosts
.....

We assume that you have root access to 3 Ubuntu 20.04 Server hosts / VMs. We recommend at least 4 CPU, 8GB RAM flavors for the base services and the reducer, and 4 CPU, 16BG RAM for the combiner host. 
You need to be able to configure security groups / ingress settings for each host.

Certificates (optional)
.......................

Certificates are needed for the reducer and each of the combiners for secure communication. 
By default, FEDn will generate unsigned certificates for the reducer and each combiner nodes using OpenSSL. 

.. note:: 
   Certificates based on IP addresses are not supported due to incompatibilities with gRPC. 

Token authentication (optional)
.......................
FEDn supports single token authentication between combiners/clients and the reducer. To enable token authentication use :code:`--secret_key=<your-key-phrase>` flag when starting the reducer.
The secret key will generate a token (expires after 90 days by default) and display it in the standard output.
Using this configuration will require combiners and clients to authenticate via either :code:`--token=<generated-token>` or by specifying the "token: <generated-token>" in the settings YAML file provided to :code:`--init`.


.. note::
   The instructions below (1-4) does not use token authentication.

Networking / DNS 
................
The reducer/controller as well as each client needs to be able to resolve the hostname given in the combiner configuration file. In this example, 
we show how this can be achieved if no external DNS lookup is available by setting "extra host" in the Docker containers for the Reducer and client.    


1. Deploy supporting services (MinIO, MongoDB and MongoExpress)  
---------------------------------------------------------------

First, deploy Minio and Mongo services on one host. Edit 'config/base-services.yaml', to change the default passwords. 

.. note::
   Remember to open ports on the host so that the reducer can reach the API endpoints (the exported port in the 'ports' properety for each of the services). 
   Note that you can also configure the reducer to use already existing MongoDB and MinIO backends so this step is optional.    

2. Deploy the reducer
---------------------

Follow the steps for pseudo-distributed deployment, but now: 
   a. Edit 'settings-reducer.yaml' to provide the appropriate connection settings for MongoDB and Minio from Step 1. 
   b. Copy 'config/extra-hosts-reducer.yaml.template' to 'config/extra-hosts-reducer.yaml' and edit it, adding a host:IP mapping for each combiner you plan to deploy. 
Then you can start the reducer: 

.. code-block:: bash

   sudo docker-compose -f config/reducer.yaml -f config/extra-hosts-reducer.yaml up 


.. note::
   Step b is a way to add the host:IP mapping to /etc/hosts in the Docker container in docker-compose. This step can be skipped if you handle this resolution in some other way. 

3. Deploy combiners
-------------------

Edit 'config/settings-combiner.yaml' to provide a name for the combiner (used as a unique identifier for the combiner in the network), 
a hostname (which is used by reducer and clients to connect to combiner RPC), and the port (default is 12080, make sure to allow access to this port in your security group/firewall settings). Also, provide the IP and port for the reducer under the 'controller' tag. Then deploy the combiner: 

.. code-block:: bash

   sudo docker-compose -f config/combiner.yaml up 

Optional: Repeat the same steps for the second combiner node. Make sure to provide unique names for the two combiners. 

> Note that it is not currently possible to use the host's IP address as 'host'. This is due to gRPC not being able to handle certificates based on IP. 

4. Attach clients to the FEDn network
-------------------------------------

Once the FEDn network is deployed, you can attach clients to it in the same way as for the pseudo-distributed deployment. 
Note that the clients need host:IP resolution for each combiner node in the network. See the reducer deployment for an 
example of how you can handle it using docker-compose. 
