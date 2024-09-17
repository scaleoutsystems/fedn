FEDn Project: 'Yolo-object-detection'
-------------------------------------------

This is a License Plate Detection version of the Darknet Quickstart Tutorial. For a step-by-step guide, refer to that tutorial.

   **Note: We recommend all new users to start by following the Quickstart Tutorial: https://fedn.readthedocs.io/en/latest/quickstart.html**

Prerequisites
-------------------------------------------

-  `Python >=3.8, <=3.12 <https://www.python.org/downloads>`__

Creating the compute package and seed model
-------------------------------------------

Install fedn: 

.. code-block::

   pip install fedn

Clone this repository, then locate into this directory:

.. code-block::

   git clone https://github.com/scaleoutsystems/fedn.git
   cd fedn/examples/Yolo-object-detection

Create the compute package:

.. code-block::

   fedn package create --path client

This should create a file 'package.tgz' in the project folder.

Next, generate a seed model (the first model in a global model trail):

.. code-block::

   fedn run build --path client

This step will take a few minutes, it also clones the darknet folder

Go into the darknet folder and edit the Makefile if you have GPU, and run make

.. code-block::

   fedn run startup --path client

This step will download the training and test dataset, creates the train.txt and val.txt inside the data folder


Running the project on FEDn
----------------------------

To learn how to set up your FEDn Studio project and connect clients, take the quickstart tutorial: https://fedn.readthedocs.io/en/stable/quickstart.html. 

