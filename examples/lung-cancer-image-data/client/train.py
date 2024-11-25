import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data import load_data
from model import load_parameters, save_parameters

from fedn.utils.helpers.helpers import save_metadata

def train(in_model_path, out_model_path, data_path=None, batch_size=40, epochs=1):
    
    # Load data
    train_set, val_set = load_data(data_path)

    image_gen = ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input)
    train = image_gen.flow_from_dataframe(dataframe= train_set,x_col="filepaths",y_col="labels",
                                          target_size=(244,244),
                                          color_mode='rgb',
                                          class_mode="categorical", #used for Sequential Model
                                          batch_size=batch_size,
                                          shuffle=False            #do not shuffle data
                                         )
    val = image_gen.flow_from_dataframe(dataframe= val_set,x_col="filepaths", y_col="labels",
                                        target_size=(244,244),
                                        color_mode= 'rgb',
                                        class_mode="categorical",
                                        batch_size=batch_size,
                                        shuffle=False
                                       )


    classes=list(train.class_indices.keys()) 



    # Load model
    model = load_parameters(in_model_path)

   

    # Train
    model.fit(train, epochs=epochs, validation_data=val, verbose=1)

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(train),
        "batch_size": batch_size,
        "epochs": epochs,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
