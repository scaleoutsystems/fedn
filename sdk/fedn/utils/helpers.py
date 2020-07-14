
import tensorflow.keras as keras
import tensorflow.keras.models as krm 
import tempfile
import os
import numpy as np

class KerasSequentialHelper:
    """ FEDn helper class for keras.Sequential. """

    def average_weights(self,models):
        """ Average weights of Keras Sequential models. """
        weights = [model.get_weights() for model in models]

        avg_w = []
        for l in range(len(weights[0])):
            lay_l = np.array([w[l] for w in weights])
            weight_l_avg = np.mean(lay_l,0)
            avg_w.append(weight_l_avg)

        return avg_w

    def increment_average(self,model,model_next,n):
        """ Update an incremental average. """
        w_prev = self.get_weights(model)
        w_next = self.get_weights(model_next)
        w = np.add(w_prev,(np.array(w_next) - np.array(w_prev))/n)
        self.set_weights(model,w)

    def set_weights(self,model,weights):
        model.set_weights(weights)

    def get_weights(self,model):
        return model.get_weights()

    def load_model(self,model):

        """ We need to go via a tmpfile to load bytestream serializd models retrieved from the miniorepository. """
        fod, outfile_name = tempfile.mkstemp(suffix='.h5') 
        with open(outfile_name,'wb') as fh:
            fh.write(model)
        model = krm.load_model(outfile_name)
        os.unlink(outfile_name)
        return model

