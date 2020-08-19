
from pandas import read_csv
import numpy as np
from os import mkdir

def pre_process(): 
    csv_path = "./raw_data.csv"

    data = np.array(read_csv(csv_path, error_bad_lines=False))[::,1]
    data = data.astype(np.float)

    sq_len = 20
    result = []
    for index in range(len(data) - sq_len):
        result.append(data[index:index+sq_len])
    result = np.array(result)

    np.savetxt("./all_data.csv",
               fmt='%.6f',X = result, delimiter=",")


def partition_data(filename, num):
    data = np.array(read_csv(filename))
    train = data[0:int(0.8*data.shape[0]), :]
    test = data[int(0.8*data.shape[0]):data.shape[0], :]

    length = int(train.shape[0]/num)
    for i in range(num):
        mkdir("./data"+str(i+1))
        local = train[i*length:(i+1)*length,:]
        np.savetxt("./data"+str(i+1)+"/train.csv", fmt='%.6f', X=local, delimiter=",")
        np.savetxt("./data"+str(i+1)+"/test.csv", fmt='%.6f', X=test, delimiter=",")

if __name__ == '__main__':
    pre_process()
    partition_data("./all_data.csv", 5)

