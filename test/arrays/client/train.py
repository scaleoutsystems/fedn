import numpy as np
import pickle
import sys

if __name__ == '__main__':

    from fedn.utils.numpymodel import NumpyHelper
    helper = NumpyHelper()
    model = helper.load_model(sys.argv[1])
    u = np.random.random()
    if u <= 0.5:
        model = np.ones(np.shape(model))
    else: 
        model = 2.0*np.ones(np.shape(model))

    helper.save_model(model,sys.argv[2])