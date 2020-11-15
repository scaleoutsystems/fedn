import numpy
import pickle
import json
import sys

if __name__ == '__main__':

    from fedn.utils.numpymodel import NumpyHelper
    helper = NumpyHelper()
    model = helper.load_model(sys.argv[1])
 
    data = numpy.ones(numpy.shape(model))
    report = {'accuracy': numpy.sum(model-data)}

    with open(sys.argv[2],"w") as fh:
        fh.write(json.dumps(report))