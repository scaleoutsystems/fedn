import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import os
import sys
sys.path.append('/Users/mattiasakesson/Documents/projects/fedn/fedn')
from fedn.utils.kerasweights import KerasWeightsHelper
from ..client.keras_model_structure import create_seed_model



if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = create_seed_model()
	outfile_name = "879fa112-c861-4cb1-a25d-775153e5b548"

	weights = model.get_weights()
	helper = KerasWeightsHelper()
	helper.save_model(weights, 'weights.npz')
