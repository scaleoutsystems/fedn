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


## Default/suggested client structure
We recommend the following project structure for a client implementation: 

## Training entrypoint
![alt-text](img/TrainSISO.png?raw=true "Training entrypoint")

The above figure gives a logical overview of the role of the training entrypoint. It should be a single-input single-putput program, taking as input a model update file and writing a model update file (same file format). Staging and upload of these files are handled by the FEDn client. A helper class in the FEDn SDK handled the ML-framework specific file serialization and deserialization. 

Below is the content of the default mnist-keras example train.py. Note how the helper is used in main to read the model from a filename given on stdin, and to write the model to a file after training. 

```python
from __future__ import print_function
import sys
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as krm
import numpy as np
import pickle
import yaml
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from data.read_data import read_data
import os


def train(model,data,settings):
    print("-- RUNNING TRAINING --", flush=True)

    # We are caching the partition in the container home dir so that
    # the same training subset is used for each iteration for a client.
    try:
        x_train = np.load('/tmp/local_dataset/x_train.npz')
        y_train = np.load('/tmp/local_dataset/y_train.npz')
    except:
        (x_train, y_train, classes) = read_data(data,
                                                nr_examples=settings['training_samples'],
                                                trainset=True)

        try:
            os.mkdir('/tmp/local_dataset')
            np.save('/tmp/local_dataset/x_train.npz',x_train)
            np.save('/tmp/local_dataset/y_train.npz',y_train)

        except:
            pass

    model.fit(x_train, y_train, batch_size=settings['batch_size'], epochs=settings['epochs'], verbose=1)

    print("-- TRAINING COMPLETED --", flush=True)
    return model

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
## Data access 
There are many possible ways to interact with the local dataset. In principle, the only requirement is that the train and validate endpoints are able to correctly read and use the data. In practice, it is then necessary to make some assumption on the local environemnt when writing train.py and validate.py. This is best explained by looking at the code above. Here we assume that the dataset is present in a file called "mnist.npz" in a folder "data" one level up in the file hierarchy relative to the exection of train.py. Then, independent on the preferred way to run the client (native, Docker, K8s etc) this structure needs to be maintained for this particular compute package. Note however, that there are many ways to accompish this on a local operational level.  



## Providing the runtime environment
