import os

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def load_data(data_path=None, is_train=True):
    """Load data from data_path. If data_path is None, load data from default path."""
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/clients/1/diabetes.pt")

    data = torch.load(data_path, weights_only=True)
    if is_train:
        return data["X_train"]
    else:
        return data["X_test"]


def load_labels(data_path=None, is_train=True):
    """Load labels from data_path. If data_path is None, load labels from default path."""
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/clients/labels.pt")
    data = torch.load(data_path, weights_only=True)
    if is_train:
        return data["y_train"]
    else:
        return data["y_test"]


def vertical_split(out_dir="data", n_splits=2):
    """Generate vertical splits for diabetes dataset for n_splits clients."""
    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.makedirs(f"{out_dir}/clients")

    df_diabetes = pd.read_csv("../data/diabetes.csv")

    # data preprocessing
    df_diabetes[["Glucose", "BloodPressure", "SkinThickness", "BMI"]] = df_diabetes[["Glucose", "BloodPressure", "SkinThickness", "BMI"]].replace(0, np.nan)
    imputer = SimpleImputer(strategy="mean")
    df_diabetes[["Glucose", "BloodPressure", "SkinThickness", "BMI"]] = imputer.fit_transform(df_diabetes[["Glucose", "BloodPressure", "SkinThickness", "BMI"]])

    y = df_diabetes["Outcome"].to_numpy()
    X = df_diabetes.drop(columns=["Outcome"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # vertical data split
    features_1 = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness"]
    features_2 = ["Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    # features_1 = ["Glucose", "BloodPressure"]
    # features_2 = ["Insulin", "BMI"]

    X_train_1 = X_train[features_1]
    X_train_2 = X_train[features_2]

    X_test_1 = X_test[features_1]
    X_test_2 = X_test[features_2]

    # scaling
    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()

    # train
    X_train_1_scaled = scaler_1.fit_transform(X_train_1)
    X_train_2_scaled = scaler_2.fit_transform(X_train_2)

    # test
    X_test_1_scaled = scaler_1.transform(X_test_1)
    X_test_2_scaled = scaler_2.transform(X_test_2)

    # to tensor
    X_train_1_tensor = torch.tensor(X_train_1_scaled, dtype=torch.float32)
    X_train_2_tensor = torch.tensor(X_train_2_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_test_1_tensor = torch.tensor(X_test_1_scaled, dtype=torch.float32)
    X_test_2_tensor = torch.tensor(X_test_2_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    data = {
        "train_features": [X_train_1_tensor, X_train_2_tensor],
        "train_labels": y_train_tensor,
        "test_features": [X_test_1_tensor, X_test_2_tensor],
        "test_labels": y_test_tensor,
    }

    # create vertical splits
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i + 1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        # save features
        torch.save(
            {
                "X_train": data["train_features"][i],
                "X_test": data["test_features"][i],
            },
            f"{subdir}/diabetes.pt",
        )
    # save labels
    subdir = f"{out_dir}/clients"
    torch.save(
        {
            "y_train": data["train_labels"],
            "y_test": data["test_labels"],
        },
        f"{subdir}/labels.pt",
    )


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        vertical_split()
