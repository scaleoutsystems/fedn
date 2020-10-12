import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import os


import tempfile

# Create an initial CNN Model
def create_seed_model():
	# input image dimensions
	img_rows, img_cols = 28, 28
	input_shape = (img_rows, img_cols, 1)
	num_classes = 10

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
    	          optimizer=keras.optimizers.Adadelta(),
        	      metrics=['accuracy'])
	return model

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = create_seed_model()
	outfile_name = "879fa112-c861-4cb1-a25d-775153e5b548"
#	fod, outfile_name = tempfile.mkstemp(suffix='.h5') 
	model.save(outfile_name)

	#project = Project()
	#from scaleout.repository.helpers import get_repository

	#repo_config = {'storage_access_key': 'minio',
#				   'storage_secret_key': 'minio123',
#				   'storage_bucket': 'models',
#				   'storage_secure_mode': False,
#				   'storage_hostname': 'minio',
#				   'storage_port': 9000}
#	storage = get_repository(repo_config)

#	model_id = storage.set_model(outfile_name,is_file=True)
#	os.unlink(outfile_name)
#	print("Created seed model with id: {}".format(model_id))

	


import os

from scaleout.project import Project

import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Create an initial CNN Model
def create_seed_model():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.mp = nn.MaxPool2d(2)

            self.fc = nn.Linear(320, 10)

        def forward(self, x):
            in_size = x.size(0)
            x = F.relu(self.mp(self.conv1(x)))
            x = F.relu(self.mp(self.conv2(x)))
            x = x.view(in_size, -1)
            x = self.fc(x)
            return F.log_softmax(x)

    model = Net()
    optimizer = optim.Adadelta(model.parameters())
    loss = F.nll_loss
    return model, optimizer, loss


if __name__ == '__main__':

# Create a seed models and push to Minio
	model, optimizer, _ = create_seed_model()
	torch.save({'model_state_dict': model.state_dict()}, "879fa112-c861-4cb1-a25d-775153e5b548")
