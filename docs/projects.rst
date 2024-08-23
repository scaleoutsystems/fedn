.. _projects-label:

================================================
Develop your own FEDn project
================================================

This guide explains how a FEDn project is structured, and details how to develop and run your own
projects. 

**In this article**
`Prerequisites`_
`Overview`_
`Build a FEDn project`_
`Deploy a FEDn project`_

Prerequisites
==============


 
Overview
==========

A FEDn project is a convention for packaging/wrapping machine learning code to be used for federated learning with FEDn. At the core, 
a project is a directory of files (often a Git repository), containing your machine learning code, FEDn entry points, and a specification 
of the runtime environment (python environment or a Docker image). The FEDn API and command-line tools provides functionality
to help a user automate deployment and management of a project that follows the conventions. 



Build a FEDn project
=====================

We recommend that projects have roughly the following folder and file structure:

| project
| ├ client
| │   ├ fedn.yaml
| │   ├ python_env.yaml
| │   ├ model.py
| │   ├ data.py
| │   ├ train.py
| │   └ validate.py
| ├ data
| │   └ mnist.npz
| ├ README.md
| ├ scripts / notebooks
| └ Dockerfile / docker-compose.yaml
|

The ``client`` folder is commonly referred to as the *compute package* and it contains files with logic specific to a single client. The file ``fedn.yaml`` is the FEDn Project File and contains information about the commands that fedn will run when a client recieves a new train or validation request. These fedn commmands are referred to as ``entry points`` and there are up to four entry points in the project folder example given above that need to be specified, namely: 
**build** - used for any kind of setup that needs to be done before the client starts up, such as initializing the global seed model. In the `quickstart tutorial<https://fedn.readthedocs.io/en/stable/quickstart.html>`_, it runs model.py when called
**startup** - used immediately after the client starts up and the environment has been initalized. In the `quickstart tutorial<https://fedn.readthedocs.io/en/stable/quickstart.html>`_, it runs data.py when invoked
**train** - runs train.py when called 
**validate** - runs validate.py when called

The compute package content (client folder)
-------------------------------------------

**The Project File (fedn.yaml)**

FEDn uses a project file named 'fedn.yaml' to specify which entry points to execute when the client recieves a training or validation request, and 
what environment to execute those entry points in. 

.. code-block:: yaml

    python_env: python_env.yaml

    entry_points:
        build:
            command: python model.py
        startup:
            command: python data.py
        train:
            command: python train.py
        validate:
            command: python validate.py


**Environment (python_env.yaml)**

It is assumed that all entry points are executable within the client runtime environment. As a user, you have two main options 
to specify the environment: 

    1. Provide a ``python_env`` in the ``fedn.yaml`` file. In this case, FEDn will create an isolated virtual environment and install the project dependencies into it before starting up the client. FEDn currently supports Virtualenv environments, with packages on PyPI. 
    2. Manage the environment manually. Here you have several options, such as managing your own virtualenv, running in a Docker container, etc. Remove the ``python_env`` tag from ``fedn.yaml`` to handle the environment manually.  


**build (optional):**

This entry point is used for any kind of setup that **needs to be done before the client starts up**, such as initializing the global seed model, and is intended to be called **once**.


**startup (optional):**

Like the 'build' entry point, 'startup' is also called **once**, immediately after the client starts up and the environment has been initalized. 
It can be used to do runtime configurations of the local execution environment. For example, in the `quickstart tutorial<https://fedn.readthedocs.io/en/stable/quickstart.html>`_, 
the startup entry point invokes a script that downloads the MNIST dataset and creates a partition to be used by that client. 
This is a convenience useful for automation of experiments and not all clients will specify such a script. 


**train (mandatory):** 

This entry point is invoked every time the client recieves a new model update (training) request. The training entry point must be a single-input single-output (SISO) program. It will be invoked by FEDn as such: 

.. code-block:: python

    python train.py model_in model_out

where 'model_in' is the **file** containing the current global model to be updated, and 'model_out' is a **path** to write the new model update to.
Download and upload of these files are handled automatically by the FEDn client, the user only specifies how to read and parse the data contained in them (see `examples<https://github.com/scaleoutsystems/fedn/tree/master/examples>`_). 

The format of the input and output files (model updates) are using numpy ndarrays. A helper instance :py:mod:`fedn.utils.helpers.plugins.numpyhelper` is used to handle the serialization and deserialization of the model updates. 


**validate (optional):** 

The validation entry point is invoked every time the client recieves a validation request. It can be used to specify how a client should validate the current global
model on local test/validation data. It should read a model update from file, validate it (in any way suitable to the user), and write  a **json file** containing validation data:

.. code-block:: python

    python validate.py model_in validations.json

The validate entry point is optional. 


Deploy a FEDn project
===================

We recommend you to test your entry points locally before deploying your FEDn project. You can test *train* and *validate* by (example for the mnist-keras 
project):

.. code-block:: bash

    python train.py ../seed.npz ../model_update.npz --data_path ../data/mnist.npz
    python validate.py ../model_update.npz ../validation.json --data_path ../data/mnist.npz

You can also test *train* and *validate* entrypoint using CLI command:

.. code-block:: bash

    fedn run train --path client --input <path to input model parameters> --output <path to write the updated model parameters>
    fedn run validate --path client --input <path to input model parameters> --output <Path to write the output JSON containing validation metrics>

Note that we here assume execution in the correct Python environment. 

To deploy a project to FEDn (Studio or pseudo-local) we simply compress the compute package as a .tgz file. using fedn command line tool or manually:

.. code-block:: bash

    fedn package create --path client


The created file package.tgz can then be uploaded to the FEDn network using the :py:meth:`fedn.network.api.client.APIClient.set_package` API. FEDn then manages the distribution of the compute package to each client. 
Upon receipt of the package, a client will unpack it and stage it locally.

.. image:: img/ComputePackageOverview.png
   :alt: Compute package overview
   :width: 100%
   :align: center

The above figure provides a logical view of how FEDn uses the compute package. When the :py:mod:`fedn.network.client`  
recieves a model update or validation request, it calls upon a Dispatcher that looks up entry point definitions 
in the compute package from the FEDn Project File to determine which code files to execute. 

Before starting a training or validation session, the global seed model needs to be initialized which in our example is done by invoking the build entry point.

To invoke the build entry point using the CLI: 

.. code-block:: bash
    fedn run build --path client


More on local data access
--------------------------

There are many possible ways to interact with the local dataset. In principle, the only requirement is that the train and validate end points are able to correctly 
read and use the data. In practice, it is then necessary to make some assumption on the local environemnt when writing entrypoint.py. This is best explained 
by looking at the code above. Here we assume that the dataset is present in a file called "mnist.npz" in a folder "data" one level up in the file hierarchy relative to 
the execution of entrypoint.py. Then, independent of the preferred way to run the client (native, Docker, K8s etc) this structure needs to be maintained for this particular 
compute package. Note however, that there are many ways to accomplish this on a local operational level.


