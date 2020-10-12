import os
import tempfile

import numpy as np
import tensorflow.keras.models as krm
import torch
import collections

class KerasSequentialHelper:
    """ FEDn helper class for keras.Sequential. """

    def average_weights(self, models):
        """ Average weights of Keras Sequential models. """
        weights = [model.get_weights() for model in models]

        avg_w = []
        for l in range(len(weights[0])):
            lay_l = np.array([w[l] for w in weights])
            weight_l_avg = np.mean(lay_l, 0)
            avg_w.append(weight_l_avg)

        return avg_w

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w_prev = self.get_weights(model)
        w_next = self.get_weights(model_next)
        w = np.add(w_prev, (np.array(w_next) - np.array(w_prev)) / n)
        self.set_weights(model, w)

    def set_weights(self, model, weights):
        model.set_weights(weights)

    def get_weights(self, model):
        return model.get_weights()

    def save_model(self, model, path):
        model.save(path)

    def load_model(self, model):
        """ We need to go via a tmpfile to load bytestream serializd models retrieved from the miniorepository. """
        fod, outfile_name = tempfile.mkstemp(suffix='.h5')
        with open(outfile_name, 'wb') as fh:
            s = fh.write(model)
            print("Written {}".format(s),flush=True)
            fh.flush()
        fh.close()
        model = krm.load_model(outfile_name)
        os.unlink(outfile_name)
        return model


class PytorchModelHelper:
    def average_weights(self, models):
        """ Average weights of pytorch models. """
        weights_list = [model.state_dict() for model in models]
        return self.avg_dictionaries(weights_list)

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w_0 = self.get_weights(model)
        w_1 = self.get_weights(model_next)
        w = {key: torch.div((w_1[key] - w_0.get(key, 0)), float(n))+w_0.get(key, 0)
             for key in w_0}
        self.set_weights(model, w)

    def set_weights(self, model, weights):
        model.load_state_dic(weights, strict=True)

    def get_weights(self, model):
        return model.state_dict()

    def load_model(self, model):
        """ We need to go via a tmpfile to load bytestream serializd models retrieved from the miniorepository. """
        outfile_name = self.get_random_name()
        with open(outfile_name, 'wb') as fh:
            s = fh.write(model)
            print("Written {}".format(s),flush=True)
            fh.flush()
        fh.close()
        model, _, _ = create_seed_model()
        checkpoint = torch.load(outfile_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        os.unlink(outfile_name)
        return model

    def save_model(self, model, path):
        torch.save({'model_state_dict': model.state_dict()}, path)

    def avg_dictionaries(self, dics):
        print("Averaging weights... length of dics", len(dics))
        result = collections.OrderedDict()
        while bool(dics[0]):
            sum = 0
            for i in range(0, len(dics)):
                key, value = dics[i].popitem()
                sum += np.array(value)
            result[key] = torch.from_numpy(sum / len(dics))
        return collections.OrderedDict(reversed(list(result.items())))
