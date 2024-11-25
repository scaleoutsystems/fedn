import os
import sys

import numpy as np
from data import load_data
from model import load_parameters
import tensorflow as tf
from fedn.utils.helpers.helpers import save_metrics

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def validate(in_model_path, out_json_path, data_path=None):
    
    # Load data
    train_set, val_set = load_data(data_path)
    test_images = load_data(data_path, is_train=False)

    image_gen = ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input)
    test = image_gen.flow_from_dataframe(dataframe= test_images,x_col="filepaths", y_col="labels",
                                         target_size=(244,244),
                                         color_mode='rgb',
                                         class_mode="categorical",
                                         batch_size=4,
                                         shuffle= False
                                        )
    val = image_gen.flow_from_dataframe(dataframe= val_set,x_col="filepaths", y_col="labels",
                                        target_size=(244,244),
                                        color_mode= 'rgb',
                                        class_mode="categorical",
                                        batch_size=4,
                                        shuffle=False
                                       )

    # Load model
    model = load_parameters(in_model_path)


    # Evaluate
    model_score = model.evaluate(val, verbose=1)
    model_score_test = model.evaluate(test, verbose=1)
    y_pred = model.predict(test)
    y_pred = np.argmax(y_pred, axis=1)


    print('model_score: ', model_score)
    print('model_score_test: ', model_score_test)
    print('y_pred: ', y_pred)

    # JSON schema
    report = {
        "training_loss": model_score[0],
        "training_accuracy": model_score[1],
        "test_loss": model_score_test[0],
        "test_accuracy": model_score_test[1],
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
