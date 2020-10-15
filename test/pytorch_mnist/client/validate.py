import sys

import torch
from read_data import read_data, read_model

import json


def validate(model, loss_func, path, sample_fraction=1):
    
    try:
        test_x, test_y = read_data(path,sample_fraction=sample_fraction)
        output = model(test_x)
        test_loss = loss_func(output, test_y, size_average=False)
        max_vals, max_indices = torch.max(output, 1)
        correct = (max_indices == test_y).sum(dtype=torch.float32)
        test_loss /= len(test_y)
        accuracy = correct / len(test_y)
        result = open("../result/result.txt", "a")
        result.write("===========================\n")
        result.write("Validation accuracy: %s\n"%accuracy)
        result.close()
        print("======================================================")
        print('Validation loss:', test_loss)
        print('Validation accuracy:', accuracy)
    except Exception as e:
        print("failed to validate the models {}".format(e),flush=True)
        raise
    
    report = { 
                "classification_report": correct,
                "loss": test_loss,
                "accuracy": accuracy
            }

    return report



if __name__ == '__main__':

    model, _, loss = read_model(sys.argv[1])
    report = validate(model, loss, '../data/test.csv')

    with open(sys.argv[2],"w") as fh:
        fh.write(json.dumps(report))
