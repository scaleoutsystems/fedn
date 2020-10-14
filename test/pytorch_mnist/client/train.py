from __future__ import print_function
import sys
from random import sample
import torch


from read_data import read_data, read_model

def train(model, optimizer, loss_func, path, sample_fraction):
    # Read training data
    data, target = read_data(path,sample_fraction=sample_fraction)
    epoch=1
    # idx: Number of trained samples
    idx = 0
    batch_size = 32

    while idx < epoch*target.shape[0]:
        idx += batch_size
        samples = sample(range(target.shape[0]), batch_size)
        train_x = data[samples, :]
        train_y = target[samples]
        optimizer.zero_grad()
        output = model(train_x)
        loss = loss_func(output, train_y)
        loss.backward()
        optimizer.step()
    print("-- TRAINING COMPLETED --")
    return model


if __name__ == '__main__':
    model, optimizer, lossfunc = read_model(sys.argv[1])
    model = train(model, optimizer, lossfunc, '../data/train.csv',sample_fraction=0.1)
    path = sys.argv[2]
    torch.save({'model_state_dict': model.state_dict()}, path)


