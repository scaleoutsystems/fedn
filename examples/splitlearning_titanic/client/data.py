import os

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def load_data(data_path=None, is_train=True):
    """Load data from data_path. If data_path is None, load data from default path."""
    if data_path is None:
        data_path = abs_path + "/data/clients/1/titanic.pt"
    data = torch.load(data_path, weights_only=True)
    if is_train:
        return data["X_train"]
    else:
        return data["X_test"]

def load_labels(data_path=None, is_train=True):
    """Load labels from data_path. If data_path is None, load labels from default path."""
    if data_path is None:
        data_path = abs_path + "/data/clients/labels.pt"
    data = torch.load(data_path, weights_only=True)
    if is_train:
        return data["y_train"]
    else:
        return data["y_test"]

def preprocess_data(df: pd.DataFrame, scaler=None, is_train=True):
    """Preprocess data. If scaler is None, fit scaler on data. If is_train is False, remove labels from features."""
    if is_train:
        prep_df = df[["PassengerId", "Survived", "Pclass", "Sex", "Age", "Fare"]].copy() # select relevant features
    else:
        prep_df = df[["PassengerId", "Pclass", "Sex", "Age", "Fare"]].copy() # Survived should not be in test set
    # fill nas
    prep_df["Age"] = prep_df["Age"].fillna(prep_df["Age"].median())
    prep_df["Fare"] = prep_df["Fare"].fillna(prep_df["Fare"].median())

    # scale data
    if is_train:
        scaler = StandardScaler()
        prep_df[["Age", "Fare"]] = scaler.fit_transform(prep_df[["Age", "Fare"]])
    else:
        prep_df[["Age", "Fare"]] = scaler.transform(prep_df[["Age", "Fare"]])

    # categorization
    prep_df["Sex"] = prep_df["Sex"].astype("category").cat.codes
    prep_df["Pclass"] = prep_df["Pclass"].astype("category").cat.codes
    return prep_df, scaler


def vertical_split(out_dir="data"):
    """Generate vertical splits for titanic dataset for 2 clients. Hardcoded for now."""
    n_splits = 2

    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.makedirs(f"{out_dir}/clients")

    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")

    train_df, scaler = preprocess_data(train_df, is_train=True)
    test_df, _ = preprocess_data(test_df, scaler=scaler, is_train=False)

    # vertical train data split (for 2 clients, hardcoded)
    client_1_data_tensor = torch.tensor(train_df[["Sex", "Age"]].values, dtype=torch.float32)
    client_2_data_tensor = torch.tensor(train_df[["Pclass", "Fare"]].values, dtype=torch.float32)
    # labels, will only be accessed by server
    train_label_tensor = torch.tensor(train_df[["Survived"]].values, dtype=torch.float32)

    # vertical test data split (for 2 clients, hardcoded)
    test_client_1_tensor = torch.tensor(test_df[["Sex", "Age"]].values, dtype=torch.float32)
    test_client_2_tensor = torch.tensor(test_df[["Pclass", "Fare"]].values, dtype=torch.float32)
    # test labels, need to be loaded separately
    test_label_df = pd.read_csv("../data/labels.csv")
    test_label_tensor = torch.tensor(test_label_df.values, dtype=torch.float32)

    data = {
        "train_features": [client_1_data_tensor, client_2_data_tensor],
        "train_labels": train_label_tensor,
        "test_features": [test_client_1_tensor, test_client_2_tensor],
        "test_labels": test_label_tensor,
    }

    # Make 2 vertical splits
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        # save features
        torch.save(
            {
                "X_train": data["train_features"][i],
                "X_test": data["test_features"][i],
            },
            f"{subdir}/titanic.pt",
        )
    # save labels
    subdir = f"{out_dir}/clients"
    torch.save(
        {
            "y_train": data["train_labels"],
            "y_test": data["test_labels"],
        },
        f"{subdir}/labels.pt"
    )

if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        vertical_split()
