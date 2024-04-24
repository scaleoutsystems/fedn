import os
import requests
import tarfile
from pathlib import Path
import torch
import numpy as np   
from math import floor


dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def get_data(out_dir='data'):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = f"{out_dir}/aclImdb_v1.tar.gz"

    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.raw.read())
        print("data download completed")
    else:
        print("could not download data")

    # unzip file
    with tarfile.open(save_path, "r:gz") as tar:
        tar.extractall(path=out_dir)
        print("data extraction completed")

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

def load_data(data_path):
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path+'/data/clients/1/imdb.pt')
    
    data = torch.load(data_path)

    train_texts = list(data['train_texts'])
    train_labels = list(data['train_labels'])
    test_texts = list(data['test_texts'])
    test_labels = list(data['test_labels'])

    return train_texts, train_labels, test_texts, test_labels


def splitset(dataset, parts):
    n = len(dataset)
    local_n = floor(n/parts)
    result = []
    for i in range(parts):
        result.append(dataset[i*local_n: (i+1)*local_n])
    return result


def split(out_dir='data', n_splits=2):
    # Make dir
    if not os.path.exists(f'{out_dir}/clients'):
        os.makedirs(f'{out_dir}/clients')

    # Load and convert to dict
    train_texts, train_labels = read_imdb_split(f'{out_dir}/aclImdb/train')
    test_texts, test_labels = read_imdb_split(f'{out_dir}/aclImdb/test')

    # Shuffle train data
    perm = np.random.permutation(len(train_texts))
    train_texts = np.array(train_texts)[perm]
    train_labels = np.array(train_labels)[perm]
    # shuffle test data
    perm = np.random.permutation(len(train_texts))
    test_texts = np.array(test_texts)[perm]
    test_labels = np.array(test_labels)[perm]

    # Reduce data size
    train_texts = train_texts[:500]    
    train_labels = train_labels[:500]
    test_texts = test_texts[:50]
    test_labels = train_labels[:50]

    data = {
        'train_texts': splitset(train_texts, n_splits),
        'train_labels': splitset(train_labels, n_splits),
        'test_texts': splitset(test_texts, n_splits),
        'test_labels': splitset(test_labels, n_splits),
    }

    # Make splits
    for i in range(n_splits):
        subdir = f'{out_dir}/clients/{str(i+1)}'
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save({
            'train_texts': data['train_texts'][i],
            'train_labels': data['train_labels'][i],
            'test_texts': data['test_texts'][i],
            'test_labels': data['test_labels'][i],
        },
        f'{subdir}/imdb.pt')


if __name__ == '__main__':
    # Prepare data if not already done
    if not os.path.exists(abs_path+'/data/clients/1'):
        get_data()
        split()