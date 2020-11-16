import numpy
import pickle
import sys

if __name__ == '__main__':
    model = numpy.ones((100000,1))
    #helper = KerasSequentialHelper()
    #model = helper.load_model(sys.argv[1])
    #model = train(model,'../data/train.csv',sample_fraction=0.001)
    #helper.save_model(model,sys.argv[2])
    #with open('arrayseed','wb') as fh:
    #    fh.write(pickle.dumps(model))
    numpy.savetxt("arrayseed",model)