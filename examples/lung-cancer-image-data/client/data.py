import os
from math import floor
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


NUM_CLASSES = 2


def load_data(data_path, is_train=True):


    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", "/home/salman/LungCancer/image-data/IQ-OTHNCCD/")

    print('data_path:', data_path)

    benign_dirs = [data_path+r'/Bengin cases']
    Malignant_dir = [data_path+r'/Malignant cases']
    Normal_dirs = [data_path+r'/Normal cases']

    print('benign_dirs:', benign_dirs)
    print('Malignant_dir:', Malignant_dir)
    print('Normal_dirs:', Normal_dirs)
    
    filepaths = []
    labels = []
    dict_lists = [benign_dirs, Malignant_dir, Normal_dirs]
    class_labels = ['benign', 'Malignant', 'Normal']

    for i, dir_list in enumerate(dict_lists):
        for j in dir_list:
            flist = os.listdir(j)
            for f in flist:
                fpath = os.path.join(j, f)
                filepaths.append(fpath)
                labels.append(class_labels[i])

    Fseries = pd.Series(filepaths, name="filepaths")
    Lseries = pd.Series(labels, name="labels")
    Lung_data = pd.concat([Fseries, Lseries], axis=1)
    Lung_df = pd.DataFrame(Lung_data)
     
    Lung_df.shape

    train_images, test_images = train_test_split(Lung_df, test_size=0.3, random_state=42)
    train_set, val_set = train_test_split(Lung_df, test_size=0.2, random_state=42)
   
    if is_train: 
        return train_set, val_set
    else:
        return test_images

if __name__ == "__main__":

    load_data(data_path = None)
