import os
from math import floor
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


NUM_CLASSES = 2


def load_data(data_path, is_train=True):


    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", "/home/salman/")

    print('data_path:', data_path)

    df = pd.read_csv(data_path+'./survey lung cancer.csv')
     
    df['GENDER'].replace({'M': 0}, inplace=True)
    df['GENDER'].replace({'F': 1}, inplace=True)
    df['AGE'] = (df['AGE'] - df['AGE'].min()) / (df['AGE'].max() - df['AGE'].min())
    df['LUNG_CANCER'].replace({'NO': 0}, inplace=True)
    df['LUNG_CANCER'].replace({'YES': 1}, inplace=True) 

    labels = df['LUNG_CANCER']
    df = df.drop(columns='LUNG_CANCER')

    labels_arr = labels.to_numpy()
    df_arr = df.to_numpy()

    df_arr = torch.tensor(df_arr, dtype=torch.float32)
    labels_arr = torch.tensor(labels_arr, dtype=torch.float32).unsqueeze(1)

    X_train, X_test, y_train, y_test = train_test_split(df_arr, labels_arr) 
   
    if is_train: 
        return X_train, y_train
    else:
        return X_test, y_test

if __name__ == "__main__":

    load_data(data_path = None)
