from fedn.utils.pytorchhelper import PytorchHelper
from models.mnist_pytorch_model import create_seed_model



if __name__ == '__main__':

	# Create a seed model and push to Minio
	model, _, _ = create_seed_model()
	outfile_name = "../seed/seed.npz"
	helper = PytorchHelper()
	helper.save_model(model.state_dict(), outfile_name)

