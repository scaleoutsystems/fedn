import sys
sys.path.insert(0,'../../..')
from fedn.fedn.utils.kerasweights import KerasWeightsHelper
from test.mnist.client.keras_model_structure import create_seed_model



if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = create_seed_model()
	outfile_name = "879fa112-c861-4cb1-a25d-775153e5b548"

	weights = model.get_weights()
	helper = KerasWeightsHelper()
	helper.save_model(weights, 'weights.npz')
