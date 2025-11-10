.. _local_compute-label:

Local Compute: Run Scaleout Edge Clients Directly in Files
==========================================================

Scaleout Edge local compute allows you to run a client directly in a file, 
without the need of creating a compute package and uploading it to the Scaleout Edge server.
Instead, the script remains locally on the client. It defines all code necessary for federated training and instantiates
a client running in the local file. When a training session is started, the client uses the defined training logic to produce model updates. 

With Scaleout Edge local compute, you can run Scaleout Edge clients in Python scripts or notebooks. You could for example run a client on Google Colab. 
This can be useful for fast development and testing.
Follow the steps in `this notebook <https://github.com/scaleoutsystems/scaleout-client/blob/master/examples/api-tutorials/Local_Compute_Example.ipynb>`_ 
to get started. 


Steps involved
--------------
- Create a seed model and upload it to Scaleout Edge
- In your script, define all code needed for client-side training (and validation) 
- Connect the client to the Scaleout Edge project by running it locally in your script
- Start a training session through Scaleout Edge




