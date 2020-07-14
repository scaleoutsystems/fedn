import keras
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import requests
import os
import pickle
from fedn.client import AllianceRuntimeClient
from scaleout.repository.helpers import get_repository

from keras.optimizers import SGD

# Create an initial CNN Model
def create_seed_model():
        
        model = Sequential()

        model.add(Dense(32, input_dim=4, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(1))
        
        opt = SGD(lr=0.0001)
        model.compile(loss = "mae", optimizer = opt,metrics=['mae'])
        return model


if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = create_seed_model()
	runtime = AllianceRuntimeClient()
	model_id = runtime.set_model(pickle.dumps(model))
	print("Created seed model with id: {}".format(model_id))
	

