# Creating a federated model for use with FEDn 

This tutorial explains the role of the *model initiator* and walks you through the steps needed to create a federated model and deploy it to a FEDn network. 

## A FEDn project







Explore our other example models, and use them as templates to create your own project. 

The simplest way to take your keras project to an FEDn federation is to copy paste one of the test examples in the test folder.

- Replace the files in the dataset folder to the dataset for your project.
- Update the script: client/read_data.py so it imports your data correctly.
- Review the scripts: client/train.py and client/validate.py so it fits your requierments
- Replace the keras model in the seed/init_project.py to your keras model and run the script to produce a seed-model.
- Make a tar.gz archive of the client folder and copy it into the package folder: tar -czvf <project-name>.tar.gz client
 
You can now follow the steps in _Standalone deployment_ and _Train a federated model_ to test and train your FEDn implementation.
