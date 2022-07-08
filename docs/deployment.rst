Deployment
======================

This guide serves as reference deployment for setting up a FEDn network consisting of:
   -  One host/VM serving the supporting services (MinIO, MongoDB)
   -  One host/VM serving the controller / reducer 
   -  One host/VM running a combiner 
   
By running additional combiners we gain resilence, are able to scale horizontally to accomodate large number of clients, and we can achieve a decentralized network (see the architecture section of the documentation). Attaching additional combiners to an existing network is straigtforward, so we recommend to start simple and expand the network as needed.     
   
.. note:: 
   In this guide we will deploy using the provived docker-compose templates. Please note that additional configurations would be needed for a production-grade network.    

Prerequisites 
-------------

Hosts / VMs
...........

We assume that you have root access to 3 Ubuntu 20.04 Server hosts / VMs. We recommend at least 4 CPUs and 8GB RAM for the base services and the reducer, and 4 CPUs and 16BG RAM for the combiner host. Each host needs the following: 

- `Docker <https://docs.docker.com/get-docker>`_
- `Docker Compose <https://docs.docker.com/compose/install>`_
- `Python 3.8 <https://www.python.org/downloads>`_


Certificates (optional)
.......................

Certificates are needed for the Reducer and each of the Combiners to enable SSL for secure communication. 
By default, FEDn will generate unsigned certificates for the reducer and each combiner using OpenSSL, but you can 
also provide your own certificates. 

.. note:: 
   Certificates based on IP addresses are not supported due to incompatibilities with gRPC. 

Token authentication (optional)
...............................
FEDn supports token-based authentication between combiners/clients and the reducer. To enable token authentication use :code:`--secret-key=<your-key-phrase>` flag when starting the reducer. The secret key will be used to generate a JWT token (expires after 90 days by default) and display it in the standard output. Use this token to autheticate combiners and clients via either :code:`--token=<generated-token>` or by specifying the "token: <generated-token>" in the settings YAML file provided to :code:`--init`.

.. note::
   The deployment instructions below  does not use token authentication.

Networking  
..........
You will need to configure security groups / ingress settings for each host. 
Also, the reducer and clients need to be able to resolve the hostname for combiners (matching the certificate). In this example 
we show how this can be achieved if no external DNS resolution is available, by setting "extra host" in the Docker containers for the Reducer and client. Note that there are many other possible ways to achieve this, depending on your setup.  

1. Deploy supporting services (MinIO, MongoDB and MongoExpress)  
--------------------------------------------------------------------

First, deploy MinIO and Mongo services on one of the hosts. Edit the `docker-compose.yaml` file to change the default passwords and ports.

.. code-block:: bash

   sudo docker-compose up -d minio mongo mongo-express

Remember to open ports on the host so that the API endpoints (the exported port in the 'ports' property for each of the services) can be reached. 
   
.. warning::
   The deployment of MinIO and MongoDB above is insecure. For a prodcution network, please set up production deployments of the base services.   

2. Deploy the reducer
---------------------

Copy the file "config/settings-reducer.yaml.template" to "config/settings-reducer.yaml", then 

a. Edit 'settings-reducer.yaml' to provide the connection settings for MongoDB and Minio from Step 1. 
b. Copy 'config/extra-hosts-reducer.yaml.template' to 'config/extra-hosts-reducer.yaml' and edit it, adding a host:IP mapping for each combiner you plan to deploy. 

Then start the reducer: 

.. code-block:: bash

   sudo docker-compose \
      -f docker-compose.yaml \
      -f config/reducer-settings.override.yaml \
      -f config/extra-hosts-reducer.yaml \
      up -d reducer

.. note::
   the use of 'extra-hosts-reducer.yaml' is a way to add the host:IP mapping to /etc/hosts in the Docker container in docker-compose. It can be skipped if you handle DNS resolution in some other way. 

3. Deploy combiners
-------------------

Copy 'config/settings.yaml.template' to 'config/settings-combiner.yaml' and edit it to provide a name for the combiner (used as a unique identifier for the combiner in the FEDn network), a hostname (which is used by reducer and clients to connect to the combiner RPC server), 
and the port (default is 12080, make sure to allow access to this port in your security group/firewall settings). 
Also, provide the IP and port for the reducer under the 'controller' tag. Then deploy the combiner: 

.. code-block:: bash

   sudo docker-compose \
      -f docker-compose.yaml \
      -f config/combiner-settings.override.yaml \
      up -d combiner

Optional: Repeat this step for any number of additional combiner nodes. Make sure to provide an unique name for each combiner,
and update extra_hosts for the reducer (you need to restart the reducer to do so). 

.. warning:: 
   Note that it is not possible to use the IP address as 'host'. gRPC does not support certificates based on IP addresses. 

4. Attach clients to the FEDn network
-------------------------------------

You can now choose an example, upload a compute package and an initial model, and attach clients. 

- `Examples <https://github.com/scaleoutsystems/fedn/tree/master/examples>`__

.. note:: 
   The clients will also need to be able to resolve each combiner node usign the 'host' argument in the combiner settings file. 
   There is a template in 'config/extra-hosts-client.yaml.template' that can be modified for this purpose. 
