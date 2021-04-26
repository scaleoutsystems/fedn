# Creating a federated model for use with FEDn 

This tutorial walks you through the key step done by the *model initiator* when creating a federated project. [More about the key roles involved in the lifecycle of a federated project](roles.md). The example is based on the Kaggle project [Credit Card Fraud Detection: Anonymized credit card transactions labeled as fraudulent or genuine](https://www.kaggle.com/mlg-ulb/creditcardfraud). The task is to predict if a transaction is fradulent (1) or normal (0), based on 28 features (principal components). Our federated model will be an auto-encoder where the encoder is built with a simple ANN-model.   

## Prerequisites

Install fedn:

```bash 
pip install -e git://github.com/scaleoutsystems/fedn.git@master#egg=fedn\&subdirectory=fedn
```

## Creating the compute package 
The *compute package* is a bundle of the code to be executed by a data-provider/client. There only formal requirements on the compute package is that it defines a training entrypoint and a validation entrypoint. This also naturally involves relevant code to read local data. By default, [the fedn client dispatcher](client.md) will assume that the following SISO programs can be executed from the root of the compute package:   

```
python train.py model_in model_out 
```
where the format of the input and output files (model updates) are dependent on the ML framework used. A helper class defines routines for serializaion and de-serialization of model updates. 

For validations it is a requirement that the output is valid json: 

```
python validate.py model_in validation.json 
```

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
