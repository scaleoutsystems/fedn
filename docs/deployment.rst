Deployment
======================

By running multiple combiners (aggregation servers) at separate hosts/locations we can deploy highly decentralized networks. You can choose to deploy combiners
in geographical proximity to optimize for low-latency for client subgroups. In the case of a single combiner, we recover a standard centralized 
client-server architecture. Since it is straight-forward to scale the network dynamically by attaching additional combiners, we recommend that you 
start simple and expand the network as needed.     

This example serves as reference deployment for setting up a FEDn network consisting of:
   -  One host/VM serving the supporting services (MinIOs, MongoDB)
   -  One host/VM serving the controller / reducer 
   -  One host/VM running a combiner 

We will in this example use the provided docker-compose templates to deploy the components across the three different hosts / VMs. 

Prerequisites 
-------------

Hosts / VMs
...........

We assume that you have root access to 3 Ubuntu 20.04 Server hosts / VMs. We recommend at least 4 CPUs and 8GB RAM for the base services and the reducer, 
and 4 CPUs and 16BG RAM for the combiner host. Each host needs the following: 

- `Docker <https://docs.docker.com/get-docker>`_
- `Docker Compose <https://docs.docker.com/compose/install>`_
- `Python 3.8 <https://www.python.org/downloads>`_

You can use the follwing bash script to install docker and docker-compose for Ubuntu 20.04 LTS:

Certificates (optional)
.......................

Certificates are needed for the Reducer and each of the Combiners to enable SSL for secure communication. 
By default, FEDn will generate unsigned certificates for the reducer and each combiner using OpenSSL. 

.. note:: 
   Certificates based on IP addresses are not supported due to incompatibilities with gRPC. 

Token authentication (optional)
...............................
FEDn supports single token authentication between combiners/clients and the reducer. To enable token authentication use :code:`--secret-key=<your-key-phrase>` flag when starting the reducer.
The secret key will generate a token (expires after 90 days by default) and display it in the standard output.
Using this configuration will require combiners and clients to authenticate via either :code:`--token=<generated-token>` or by specifying the "token: <generated-token>" in the settings YAML file provided to :code:`--init`.


.. note::
   The instructions below (1-4) does not use token authentication.

Networking  
..........
You will also need to be able to configure security groups / ingress settings for each host. 
The Reducer as well as each client needs to be able to resolve the hostname for each combiner (matching the certificate). In this example, 
we show how this can be achieved if no DNS resolution is available, by setting "extra host" in the Docker containers for the Reducer and client.   
Note that there are many other possible ways to achieve this, depending on your setup.  

1. Deploy base/supporting services (MinIO, MongoDB and MongoExpress)  
--------------------------------------------------------------------

First, use 'config/base-services.yaml' to deploy MinIO and Mongo services on one of the hosts. Edit the file to change the default passwords and ports.

.. code-block:: bash

   sudo -E docker-compose -f config/base-services.yaml up 

.. note::
   Remember to open ports on the host so that the API endpoints (the exported port in the 'ports' property for each of the services) can be reacheds. 
   Note that you can also configure the reducer to use already existing MongoDB and MinIO services, in that case you can skip this step.    

2. Deploy the reducer
---------------------

Copy the file "config/settings-reducer.yaml.template" to "config/settings-reducer.yaml", then 

a. Edit 'settings-reducer.yaml' to provide the connection settings for MongoDB and Minio from Step 1. 
b. Copy 'config/extra-hosts-reducer.yaml.template' to 'config/extra-hosts-reducer.yaml' and edit it, adding a host:IP mapping for each combiner you plan to deploy. 

Then start the reducer: 

.. code-block:: bash

   sudo -E docker-compose -f config/reducer.yaml -f config/extra-hosts-reducer.yaml up 


.. note::
   Step b is a way to add the host:IP mapping to /etc/hosts in the Docker container in docker-compose. This step can be skipped if you handle this resolution in some other way. 

3. Deploy combiners
-------------------

Copy 'config/settings.yaml.template' to 'config/settings-combiner.yaml' and edit it to provide a name for the combiner (used as a unique identifier for the combiner in the network), 
a hostname (which is used by reducer and clients to connect to combiner RPC), 
and the port (default is 12080, make sure to allow access to this port in your security group/firewall settings). 
Also, provide the IP and port for the reducer under the 'controller' tag. Then deploy the combiner: 

.. code-block:: bash

   sudo -E docker-compose -f config/combiner.yaml up 

Optional: Repeat this step for any number of additional combiner nodes. Make sure to provide unique names for the two combiners,
and update extra hosts for the reducer. 

.. warning:: 
   Note that it is not possible to use the IP address as 'host'. gRPC does not support certificates based on IP addresses. 

4. Attach clients to the FEDn network
-------------------------------------

You can now choose an example, upload a compute package and an initial model, and attach clients. 

- `Examples <../../examples>`__

.. note:: 
   The clients will also need to be able to resolve the hostname ('host' argument) for each combiner node in the network. 
   There is a template in 'config/extra-hosts-client.yaml.template' that can be modified for this purpose. 
