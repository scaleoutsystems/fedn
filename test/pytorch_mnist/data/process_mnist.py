import os
from pandas import read_csv
import numpy as np
from os import mkdir

## Read feature data
def get_images(imgf, n):
    f = open(imgf, "rb")
    f.read(16)
    images = []

    for i in range(n):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    return images

## Read label data
def get_labels(labelf, n):
    l = open(labelf, "rb")
    l.read(8)
    labels = []
    for i in range(n):
        labels.append(ord(l.read(1)))
    return labels

## Write csv from as-read data
def output_csv(images, labels, outf):
    o = open(outf, "w")
    for i in range(len(images)):
        o.write(",".join(str(x) for x in [labels[i]] + images[i]) + "\n")
    o.close()

## workflow for read and write csv for entire data
def csv_convert(imgf, labelf, n):
    images = get_images(imgf, n)
    labels = get_labels(labelf, n)
    output_csv(images, labels, "data.csv")

## Read entire csv, partition data and save in separated directories
def partition_data(filename, num):
    data = np.array(read_csv(filename))
    train = data[0:int(0.8*data.shape[0]), :]
    test = data[int(0.8*data.shape[0]):data.shape[0], :]

    length = int(train.shape[0]/num)
    for i in range(num):
        mkdir("./data"+str(i+1))
        local = train[i*length:(i+1)*length,:]
        np.savetxt("./data"+str(i+1)+"/train.csv", fmt='%d', X=local, delimiter=",")
        np.savetxt("./data"+str(i+1)+"/test.csv", fmt='%d', X=test, delimiter=",")

if __name__ == '__main__':
    print("Converting image data to csv...")
    csv_convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000)
    print("Partitioning data...")
    partition_data("./data.csv", 5)
    print("Done! ")

