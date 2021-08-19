# Creating a new federated model for use with FEDn 

This tutorial walks you through the key step done by the *model initiator* when setting up a federated project. [More about the key roles involved in the lifecycle of a federated project](roles.md). The example is based on the well-known MNIST example project. The task is to classify hand-written digits using a simple ANN-model.   

## Prerequisites

Install fedn:

```bash 
pip install fedn
```

## The compute package explained

![alt-text](img/ComputePackageOverview.png?raw=true "Compute package overview")

The *compute package* is a tar.gz bundle of the code to be executed by each data-provider/client. This package is uploaded to the Reducer upon initialization of the FEDN Network (along with the initial model). When a client connects to the network, it downloads and unpacks the package locally and are then ready to participate in training and/or validation. 

The logic is illustrated in the above figure. When the [FEDn client](https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/client.py) recieves a model update request from the combiner, it calls upon a Dispatcher that looks up entry point definitions in the compute package. These entrypoints define commands executed by the client to update/train or validate a model: 

```
python train.py model_in model_out 
```

where the format of the input and output files (model updates) are dependent on the ML framework used. A [helper class](https://github.com/scaleoutsystems/fedn/blob/master/fedn/fedn/utils/kerashelper.py) defines serializaion and de-serialization of model updates. 

For validations it is a requirement that the output is valid json: 

```
python validate.py model_in validation.json 
```

Typically, the actual model is defined in a small library, and does not depend on FEDn. We provide several [examples](https://github.com/scaleoutsystems/examples) for different ML frameworks.  

![alt-text](img/TrainSISO.png?raw=true "Training entrypoint")

## Default/suggested client structure
We recommend the following project structure for a client implementation: 

## Data access 
There are many possible ways to interact with the local dataset. In principle, the only requirement is that the train and validate endpoints are able to correctly read and use the data. In practice, it is then necessary to make some assumption on the local environemnt when writing train.py and validate.py. This is best explained by looking at the code. For example, in our default mnist-keras example, we assume that the dataset is present in a file called 

```python
if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.kerashelper import KerasHelper
    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    from models.mnist_model import create_seed_model
    model = create_seed_model()
    model.set_weights(weights)

    model = train(model,'../data/mnist.npz',settings)
    helper.save_model(model.get_weights(),sys.argv[2])

```

## Providing the runtime environment
