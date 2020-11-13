import numpy as np
import pickle
import sys

if __name__ == '__main__':

    from fedn.utils.numpymodel import NumpyHelper
    helper = NumpyHelper()
    model = helper.load_model(sys.argv[1])
    model = np.ones(np.shape(model))
    helper.save_model(model,sys.argv[2])