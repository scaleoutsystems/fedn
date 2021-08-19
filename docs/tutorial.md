# Creating a new federated model for use with FEDn 

This tutorial walks you through the key step done by the *model initiator* when setting up a federated project. [More about the key roles involved in the lifecycle of a federated project](roles.md). The example is based on the well-known MNIST example project. The task is to classify hand-written digits using a simple ANN-model.   

## Prerequisites

Install fedn:

```bash 
pip install fedn
```

## The compute package 

![alt-text](img/ComputePackageOverview.png?raw=true "Compute package overview")

The *compute package* is a tar.gz bundle of the code to be executed by each data-provider/client. This package is uploaded to the Reducer upon initialization of the FEDN Network (along with the initial model). When a client connects to the network, it downloads and unpacks the package locally and are then ready to participate in training and/or validation. 

The logic is illustrated in the above figure. When the [FEDn client](https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/client.py) recieves a model update request from the combiner, it calls upon a Dispatcher that looks up entry point definitions in the compute package. These entrypoints define commands executed by the client to update/train or validate a model. Typically, the actual model is defined in a small library, and does not depend on FEDn. 

The only formal requirements on the compute package is that it defines a training entrypoint and a validation entrypoint. This also naturally involves relevant code to read local data. By default, the fedn client dispatcher will assume that the following SISO programs can be executed from the root of the compute package:   

```
python train.py model_in model_out 
```
where the format of the input and output files (model updates) are dependent on the ML framework used. A [helper class](https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/utils/kerashelper.py) defines serializaion and de-serialization of model updates. 

For validations it is a requirement that the output is valid json: 

```
python validate.py model_in validation.json 
```

![alt-text](img/ComputePackageOverview.png?raw=true "Compute package overview")



In a Linux terminal, create a new folder 'fraud-detection' with the following structure:
```

```

To package the client code for deployment in FEDn: 
```bash
fedn control package
```
The package is created in the parent directory of the present working directory as a .tar.gz file. Dowload this packagage and upload it to the controller. 

## Providing the runtime environment
The compute package needs a runtime environment to execute in, and this is also specified by the model initiator. Create the files:

'fraud-detection/requirements.txt': 
```
numpy==1.18.0
tensorflow==2.3.2
pandas
keras
sklearn
```

 'fraud-detection/Dockerfile':  
```
FROM python:3.8.5
RUN pip install -e git://github.com/scaleoutsystems/fedn.git@master#egg=fedn\&subdirectory=fedn
COPY fedn-network.yaml /app/ 
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
```

## Where to go from here: 
