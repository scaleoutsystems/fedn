import keras

from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

import requests
import os
import pickle
from fedn.client import AllianceRuntimeClient
from scaleout.repository.helpers import get_repository

from keras.optimizers import SGD
import tempfile

# Create an initialize regression model

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
	fod, outfile_name = tempfile.mkstemp(suffix='.h5') 
	model.save(outfile_name)

	runtime = AllianceRuntimeClient()

	model_id = runtime.set_model(outfile_name,is_file=True)
	os.unlink(outfile_name)

	print("Created seed model with id: {}".format(model_id))

	

