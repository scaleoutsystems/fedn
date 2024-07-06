.. _projects-label:

Building your own projects
================================================

This guide explains how a FEDn project is structured, and details how to develop your own
projects for your own use-cases. 

A FEDn project is a convention for packaging/wrapping machine learning code to be used for federated learning with FEDn. At the core, 
a project is a directory of files (often a Git repository), containing your machine learning code, FEDn entry points, and a specification 
of the runtime environment (python environment or a Docker image). The FEDn API and command-line tools provides functionality
to help a user automate deployment and management of a project that follows the conventions. 
 
Overview
------------------------------

We recommend that projects have roughly the following folder and file structure:

| project
| ├ client
| │   ├ fedn.yaml
| │   ├ python_env.yaml
| │   ├ data.py
| │   ├ model.py
| │   ├ train.py
| │   └ validate.py
| ├ data
| │   └ mnist.npz
| ├ README.md
| ├ scripts / notebooks
| └ Dockerfile / docker-compose.yaml
|

The ``client`` folder is commonly referred to as the *compute package*. The file ``fedn.yaml`` is the FEDn Project File. It contains information about the ``entry points``. The entry points are used by the client to compute model updates (local training) and local validations (optional) . 
To run a project in FEDn, the client folder is compressed as a .tgz bundle and pushed to the FEDn controller. FEDn then manages the distribution of the compute package to each client. 
Upon recipt of the package, a client will unpack it and stage it locally.

.. image:: img/ComputePackageOverview.png
   :alt: Compute package overview
   :width: 100%
   :align: center

The above figure provides a logical view of how FEDn uses the compute package (client folder). When the :py:mod:`fedn.network.clients`  
recieves a model update request, it calls upon a Dispatcher that looks up entry point definitions 
in the compute package from the FEDn Project File. 

FEDn Project File (fedn.yaml)
------------------------------

FEDn uses on a project file named 'fedn.yaml' to specify which entrypoints to execute when the client recieves a training or validation request, and 
what environment to execute those entrypoints in. 

.. code-block:: yaml

    python_env: python_env.yaml

    entry_points:
        startup:
            command: python data.py
        train:
            command: python train.py
        validate:
            command: python validate.py


**Environment**

It is assumed that all entry points are executable within the client runtime environment. As a user, you have two main options 
to specify the environment: 

    1. Provide a ``python_env`` in the ``fedn.yaml`` file. In this case, FEDn will create an isolated virtual environment and install the project dependencies into it before starting up the client. FEDn currently supports Virtualenv environments, with packages on PyPI. 
    2. Manage the environment manually. Here you have several options, such as managing your own virtualenv, running in a Docker container, etc. Remove the ``python_env`` tag from ``fedn.yaml`` to handle the environment manually.  

**Entry Points**

There are up to four Entry Points to be specified.

**Build Entrypoint (build, optional):**

This entrypoint is intended to be called **once** for building artifacts such as initial seed models. However, it not limited to artifacts, and can be used for any kind of setup that needs to be done before the client starts up.

To invoke the build entrypoint using the CLI: 

.. code-block:: bash
    fedn build --



**Startup Entrypoint (startup, optional):**


This entrypoint is called **once**, immediately after the client starts up and the environment has been initalized. 
It can be used to do runtime configurations of the local execution environment. For example, in the quickstart tutorial example, 
the startup entrypoint invokes a script that downloads the MNIST dataset and creates a partition to be used by that client. 
This is a convenience useful for automation of experiments and not all clients will specify such a script. 

**Training Entrypoint (train, mandatory):** 

This entrypoint is invoked every time the client recieves a new model update request. The training entry point must be a single-input single-output (SISO) program. It will be invoked by FEDn as such: 

.. code-block:: python

    python train.py model_in model_out

where 'model_in' is the file containing the current global model to be updated, and 'model_out' is a path to write the new model update to.
Download and upload of these files are handled automatically by the FEDn client, the user only specifies how to read and parse the data contained in them (see examples) . 

**Validation Entrypoint (validate, optional):** 

The validation entry point works in a similar was as the trainig entrypoint. It can be used to specify how a client should validate the current global
model on local test/validation data. It should read a model update from file, validate it (in any way suitable to the user), and write  a **json file** containing validation data:

.. code-block:: python

    python validate.py model_in validations.json

 The validate entrypoint is optional. 

**Example train entry point**

Below is an example training entry point taken from the PyTorch getting stated project. 

.. code-block:: python

    import math
    import os
    import sys

    import torch
    from data import load_data
    from model import load_parameters, save_parameters

    from fedn.utils.helpers.helpers import save_metadata

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.abspath(dir_path))


    def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1, lr=0.01):
        """ Complete a model update.

        Load model paramters from in_model_path (managed by the FEDn client),
        perform a model update, and write updated paramters
        to out_model_path (picked up by the FEDn client).

        :param in_model_path: The path to the input model.
        :type in_model_path: str
        :param out_model_path: The path to save the output model to.
        :type out_model_path: str
        :param data_path: The path to the data file.
        :type data_path: str
        :param batch_size: The batch size to use.
        :type batch_size: int
        :param epochs: The number of epochs to train.
        :type epochs: int
        :param lr: The learning rate to use.
        :type lr: float
        """
        # Load data
        x_train, y_train = load_data(data_path)

        # Load parmeters and initialize model
        model = load_parameters(in_model_path)

        # Train
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        n_batches = int(math.ceil(len(x_train) / batch_size))
        criterion = torch.nn.NLLLoss()
        for e in range(epochs):  # epoch loop
            for b in range(n_batches):  # batch loop
                # Retrieve current batch
                batch_x = x_train[b * batch_size:(b + 1) * batch_size]
                batch_y = y_train[b * batch_size:(b + 1) * batch_size]
                # Train on batch
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                # Log
                if b % 100 == 0:
                    print(
                        f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

        # Metadata needed for aggregation server side
        metadata = {
            # num_examples are mandatory
            'num_examples': len(x_train),
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr
        }

        # Save JSON metadata file (mandatory)
        save_metadata(metadata, out_model_path)

        # Save model update (mandatory)
        save_parameters(model, out_model_path)


    if __name__ == "__main__":
        train(sys.argv[1], sys.argv[2])

        

The format of the input and output files (model updates) are using numpy ndarrays. A helper instance :py:mod:`fedn.utils.helpers.plugins.numpyhelper` is used to handle the serialization and deserialization of the model updates. 
The first function (_compile_model) is used to define the model architecture and creates an initial model (which is then used by _init_seed). The second function (_load_data) is used to read the data (train and test) from disk.  
The third function (_save_model) is used to save the model to disk using the numpy helper module :py:mod:`fedn.utils.helpers.plugins.numpyhelper`. The fourth function (_load_model) is used to load the model from disk, again
using the pytorch helper module. The fifth function (_init_seed) is used to initialize the seed model. The sixth function (_train) is used to train the model, observe the two first arguments which will be set by the FEDn client. 
The seventh function (_validate) is used to validate the model, again observe the two first arguments which will be set by the FEDn client.


Build a compute package 
--------------------------
To deploy a project to FEDn (Studio or pseudo-local) we simply compress the *client* folder as .tgz file. using fedn command line tool or manually:

.. code-block:: bash

    fedn package create --path client


The created file package.tgz can then be uploaded to the FEDn network using the :py:meth:`fedn.network.api.client.APIClient.set_package`.


More on local data access 
-------------------------

There are many possible ways to interact with the local dataset. In principle, the only requirement is that the train and validate endpoints are able to correctly 
read and use the data. In practice, it is then necessary to make some assumption on the local environemnt when writing entrypoint.py. This is best explained 
by looking at the code above. Here we assume that the dataset is present in a file called "mnist.npz" in a folder "data" one level up in the file hierarchy relative to 
the exection of entrypoint.py. Then, independent on the preferred way to run the client (native, Docker, K8s etc) this structure needs to be maintained for this particular 
compute package. Note however, that there are many ways to accompish this on a local operational level.

Testing the entry points locally
---------------------------------

We recommend you to test your entrypoints locally before uploading the compute package to Studio. You can test *train* and *validate* by (example for the mnist-keras 
project):

.. code-block:: bash

    python train.py ../seed.npz ../model_update.npz --data_path ../data/mnist.npz
    python validate.py ../model_update.npz ../validation.json --data_path ../data/mnist.npz

Note that we here assume execution in the correct Python environment. 
