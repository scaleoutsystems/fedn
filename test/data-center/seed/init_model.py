import keras
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import requests
import os


import tempfile


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
	#model = create_seed_model()
	#runtime = AllianceRuntimeClient()
	#model_id = runtime.set_model(pickle.dumps(model))
	#print("Created seed model with id: {}".format(model_id))
	
        # Create a seed model and push to Minio
        model = create_seed_model()
        outfile_name = "9908a112-c861-4cb1-a25d-775153e5543"
        #fod, outfile_name = tempfile.mkstemp(suffix='.h5')
        model.save(outfile_name)

        #project = Project()
        #from scaleout.repository.helpers import get_repository

        #repo_config = {'storage_access_key': 'minio',
        #                           'storage_secret_key': 'minio123',
        #                           'storage_bucket': 'models',
        #                           'storage_secure_mode': False,
        #                           'storage_hostname': 'minio',
        #                           'storage_port': 9000}
        #storage = get_repository(repo_config)

        #model_id = storage.set_model(outfile_name,is_file=True)
        #os.unlink(outfile_name)
        #print("Created seed model with id: {}".format(model_id)) 
        

