.. _local_compute-label:

Local Compute: Run FEDn Clients Directly in Files
=================================================

FEDn local compute allows you to run a client directly in a file, 
without the need of creating a compute package and uploading it to FEDn Studio.
Instead, the script remains locally on the client. It defines all code necessary for federated training and instantiates
a client running in the local file. When a training session is started, the client uses the defined training logic to produce model updates. 

With FEDn local compute, you can run FEDn clients in Python scripts or notebooks. You could for example run a client on Google Colab. 
This can be useful for fast development and testing.
Follow the steps in `this notebook <https://github.com/scaleoutsystems/fedn/blob/master/examples/api-tutorials/Local_Compute_Example.ipynb>`_ 
to get started. 


Steps involved
--------------
- Create a seed model and upload it to FEDn Studio
- In your script, define all code needed for client-side training (and validation) 
- Connect the client to the FEDn project by running it locally in your script
- Start a training session through FEDn Studio




