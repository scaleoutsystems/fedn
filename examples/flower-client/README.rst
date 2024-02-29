Using Flower clients in FEDn
-------------

Example of how a Flower client can be used in FEDn. Flowers quickstart-pytorch example is 
used in this example (see `flwr_client.py``). Study the `client/entrypoint` file for 
details of the implementation.
   

Run details
-----------

See `https://fedn.readthedocs.io/en/stable/quickstart.html` for general run details. Note 
that the flower client handles data distribution programatically, so data related steps can be 
omitted. To run this example after initializing fedn with the `seed.npz` and `package` that 
can be generated through `bin/build`, continue with building a docker image containing the flower 
dependencies. From the repository root execute:

.. code-block::

   docker build --build-arg REQUIREMENTS=examples/flower-client/requirements.txt -t flower-client .

In separate terminals, navigate to this folder, start clients and inject the `CLIENT_NUMBER` 
dependency, for example for client1:

.. code-block::

   docker run \
   -v $PWD/client.yaml:/app/client.yaml \
   --network=fedn_default \
   -e CLIENT_NUMBER=0 \
   flower-client run client -in client.yaml --name client1
