from fedn.utils.pytorchhelper import PytorchHelper
from models.mnist_pytorch_model import create_seed_model
import collections

def weights_to_np(weights):

    weights_np = collections.OrderedDict()
    for w in weights:
        weights_np[w] = weights[w].cpu().detach().numpy()
    return weights_np

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model, _, _ = create_seed_model()
	outfile_name = "../seed/seed.npz"
	helper = PytorchHelper()
	helper.save_model(weights_to_np(model.state_dict()), outfile_name)

