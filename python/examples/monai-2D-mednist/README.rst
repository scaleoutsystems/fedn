FEDn Project: MonAI 2D Classification with the MedNIST Dataset (PyTorch)
------------------------------------------------------------------------

This is an example FEDn Project based on the  MonAI 2D Classification with the MedNIST Dataset.
The example is intented as a minimalistic quickstart and automates the handling of training data
by letting the client download and create its partition of the dataset as it starts up.

Links:

-  MonAI: https://monai.io/
-  Base example notebook: https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb
-  MedNIST dataset: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz

Prerequisites
-------------

Using FEDn Studio:

-  `Python 3.9, 3.10 or 3.11 <https://www.python.org/downloads>`__
-  `A FEDn Studio account <https://scaleout.scaleoutsystems.com/signup>`__


Creating the compute package and seed model
-------------------------------------------

Install fedn:

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/scaleout.git
   cd fedn/examples/monai-2D-mednist

Create the compute package:

.. code-block::

   scaleout package create --path client

This should create a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in a global model trail):

.. code-block::

   scaleout run build --path client

This will create a seed model called 'seed.npz' in the root of the project. This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).

Download and Prepare the data
-------------------------------------------

Install requirements:

.. code-block::

   pip install -r requirements.txt

Download and divide the data into parts. Set the number of
data parts as an arguments python prepare_data.py NR-OF-DATAPARTS. In the
below command we divide the dataset into 10 parts.
.. code-block::

    python prepare_data.py 10


Using FEDn Studio
-----------------

Follow the guide here to set up your FEDn Studio project and learn how to connect clients (using token authentication): `Studio guide <https://scaleout.readthedocs.io/en/stable/quickstart.html>`__.
On the step "Upload Files", upload 'package.tgz' and 'seed.npz' created above.

Connecting clients:
===================

**NOTE: In case a different data path needs to be set, use the env variable SCALEOUT_DATA_PATH.**

.. code-block::

   export SCALEOUT_PACKAGE_EXTRACT_DIR=package
   export SCALEOUT_DATA_PATH=<full_path_to_the_dir>/data/
   export SCALEOUT_CLIENT_SETTINGS_PATH=<full_path_to_the_dir>/client_settings.yaml
   export SCALEOUT_DATA_SPLIT_INDEX=0

   scaleout client start -in client.yaml --secure=True --force-ssl

Connecting clients using Docker:
================================

For convenience, there is a Docker image hosted on ghrc.io with scaleout preinstalled. To start a client using Docker:

.. code-block::

   docker run \
     -v $PWD/client.yaml:/app/client.yaml \
     -v $PWD/client_settings.yaml:/app/client_settings.yaml \
     -v $PWD/data:/app/data \
     -e SCALEOUT_PACKAGE_EXTRACT_DIR=package \
     -e SCALEOUT_DATA_PATH=/app/data/ \
     -e SCALEOUT_CLIENT_SETTINGS_PATH=/app/client_settings.yaml \
     -e SCALEOUT_DATA_SPLIT_INDEX=0 \
     ghcr.io/scaleoutsystems/fedn/fedn:0.11.1 run client -in client.yaml --force-ssl --secure=True