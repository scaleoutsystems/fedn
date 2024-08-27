import os
from math import floor
import random

import numpy as np


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return np.array(result)
def write_list(array, fname):
  textfile = open(fname, "w")
  for element in array:
    textfile.write(f"{ds_path}/{element}\n")
  textfile.close()


def split(dataset="yolo_computer/data/combine_data", outdir="data", n_splits=1):
   
    #Splitting the images into train and test



    import glob, os



    current_dir = dataset


    # Percentage of images to be used for the test set

    percentage_test = 10



    # Create and/or truncate train.txt and test.txt

    file_train = open('data/train.txt', 'a')

    file_test = open('data/val.txt', 'a')

    import glob

    images_list1 = glob.glob("/home/sowmya/yolo_computer/data/combine_data/*.jpg")

    images_list2 = glob.glob("/home/sowmya/yolo_computer/data/combine_data/*.png")

    images_list3 = glob.glob("/home/sowmya/yolo_computer/data/combine_data/*.jpeg")
    images_list = images_list1 + images_list2 + images_list3


    print(images_list)

    # types = ('*.jpg', '*.png', "*jpeg")

    # Populate train.txt and test.txt

    counter = 1

    index_test = round(100 / percentage_test)

    # for pathAndFilename in glob.iglob(os.path.join(current_dir, types)):

    # title, ext = os.path.splitext(os.path.basename(pathAndFilename))


    # file = open("data/train.txt", "w")

    for id, name in enumerate(images_list):
        # file_train.write("\n".join(images_list))
        # file.close()
        if counter == index_test:
            counter = 1
            #print('in')
            file_test.write(name + "\n")
        else:
            # print('in')
            file_train.write(name + "\n")
            counter = counter + 1


def get_data(out_dir="data"):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)



if __name__ == "__main__":
    pass
    #get_data()
    #split()
