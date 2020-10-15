import numpy
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


import keras
import numpy
def read_data(filename):

    """ Helper function to read and preprocess data for training with Keras. """
    data = pd.read_csv(filename, sep = ',',index_col=[0])
    header= ['CPU', 'Power', 'Rx', 'Temprature', 'Tx']
 
    cs = MinMaxScaler()

    data[header] = cs.fit_transform(data[header])

    y = data['Power']
    X = data
    del X['Power']

    # The entire dataset is 60k images, we can subsample here for quicker testing. 
    #if sample_fraction < 1.0:
    #    foo, X, bar, y = train_test_split(X, y, test_size=sample_fraction)
    #classes = range(10)

    # The data, split between train and test sets
    return  (X, y)

