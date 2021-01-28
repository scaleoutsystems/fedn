#from fedn.utils.pytorchmodel import PytorchModelHelper
from models.mnist_pytorch_model import create_seed_model
import torch

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model, _, _ = create_seed_model()
	outfile_name = "seed.pth"
	torch.save(model.state_dict(), outfile_name)
	#helper = PytorchModelHelper()
	#helper.save_model(model.state_dict(), outfile_name)
