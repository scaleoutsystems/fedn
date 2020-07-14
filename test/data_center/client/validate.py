from __future__ import print_function
import json
import sys
import keras
import tensorflow as tf 
import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from read_data import read_data
import tensorflow.keras.models as krm



def validate(model,data):
    print("-- RUNNING Validation --")

    # The data, split between train and test sets
    (x_test, y_test) = read_data(data)


    predictions = model.predict(x_test)
    
    mse_obj = tf.keras.metrics.MeanAbsoluteError()
    mse_obj.update_state(predictions, y_test)
    mse_val =   mse_obj.result().numpy()
    
    print("-- validation COMPLETED --")

    results = {"mae" : str(mse_val)}
     
    return results

if __name__ == '__main__':
    # Read the model
    model = krm.load_model(sys.argv[1])
    validation = validate(model,'../data/test.csv')
    with open(sys.argv[2],"w") as fh:
        fh.write(json.dumps(validation))


