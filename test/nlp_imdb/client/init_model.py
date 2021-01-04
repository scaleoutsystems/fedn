from fedn.utils.kerasweights import KerasWeightsHelper
from models.imdb_model import create_seed_model

if __name__ == '__main__':
    # Create a seed model and push to Minio
    model = create_seed_model()
    outfile_name = "seed.npz"

    weights = model.get_weights()
    helper = KerasWeightsHelper()
    helper.save_model(weights, outfile_name)
