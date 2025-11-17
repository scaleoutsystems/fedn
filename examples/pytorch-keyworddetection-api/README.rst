FEDn Project: Keyword Detection (PyTorch)
-----------------------------

This is an example to showcase how to set up FEDnClient and use APIClient to setup and manage a training from python. 
The machine learning project is based on the Speech Commands dataset from Google, https://huggingface.co/datasets/google/speech_commands.

The example is intented as a minimalistic quickstart to learn how to use FEDn.


   **Note: It is recommended to complete the example in https://docs.scaleoutsystems.com/en/stable/quickstart.html before starting this example ** 

Prerequisites
-------------

-  `Python >=3.9, <=3.12 <https://www.python.org/downloads>`__
-  `A project in FEDn Studio  <https://scaleout.scaleoutsystems.com/signup>`__   

Installing pre requirements and creating seed model
-------------------------------------------

There are two alternatives to install the required packages, either using conda or pip.

.. code-block::

   conda env create -n <name-of-env> --file env.yaml

Or if you rather use pip to install the packages:

.. code-block::

   pip install -r requirements.txt

.. note::

   In the case of installing with pip need to install either sox (macos or linux) or soundfile (windows) depending on your platform as this is a requirement for the torchaudio package.

   For MacOS, you can install sox with the following command:
   .. code-block::
      brew install sox
   

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/scaleout.git
   cd fedn/examples/pytorch-keyworddetection-api

Next we need to setup the APIClient. This link https://docs.scaleoutsystems.com/en/stable/apiclient.html helps you to get the hostname and access token. Edit the file fedn_api.py and insert your HOST and TOKEN.

Next, generate the seed model:

.. code-block::

   python fedn_api.py --upload-seed

This will create a model file 'seed.npz' in the root of the project and upload it to the server.


Now we need to start the clients, download at set of client configutations following the quickstart tutorial: https://scaleout.readthedocs.io/en/stable/quickstart.html#start-clients. 

Start the clients with the following command:
.. code-block::

   python main.py --dataset-split-idx 0 --client-yaml client0.yaml

where each client is started with a different dataset split index and client yaml file.

