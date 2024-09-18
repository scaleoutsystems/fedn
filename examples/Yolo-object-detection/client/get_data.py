import os
import glob
import zipfile
import requests


dir_path = os.path.dirname(os.path.realpath(__file__))

def download_config():
    url = "https://storage.googleapis.com/public-scaleout/Yolo-object-detection/1.zip"
    response = requests.get(url)
    os.makedirs(dir_path + "/data", exist_ok=True)
    with open(dir_path + "/data/1.zip", "wb") as f:
        f.write(response.content)
    zip_file_path = dir_path + "/data/1.zip"
    extract_to_path = dir_path + "/data/"

    os.makedirs(extract_to_path, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)

def download_blob():
    url = "https://storage.googleapis.com/public-scaleout/Yolo-object-detection/data.zip"
    response = requests.get(url)

    with open(dir_path + "/data/1/data.zip", "wb") as f:
        f.write(response.content)
    zip_file_path = dir_path + "/data/1/data.zip"
    extract_to_path = dir_path + "/data/1/"

    os.makedirs(extract_to_path, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)

def train_test_creation(dataset=dir_path + "/data/1/data/", outdir=dir_path + "/data/1/", test_percentage=10):

    percentage_test = test_percentage
    file_train = open(outdir+"train.txt", "w")
    file_test = open(outdir+"val.txt", "w")
    images_list1 = glob.glob(dataset+"*.jpg")
    images_list2 = glob.glob(dataset+"*.png")
    images_list3 = glob.glob(dataset+"*.jpeg")
    images_list = images_list1 + images_list2 + images_list3

    counter = 1
    index_test = round(100 / percentage_test)

    for id, name in enumerate(images_list):
        if counter == index_test:
            counter = 1

            file_test.write(name + "\n")
        else:

            file_train.write(name + "\n")
            counter = counter + 1

def update_all_keys_in_obj_data(file_path, updates_dict):
    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(file_path, "w") as file:
        for line in lines:
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                if key in updates_dict:
                    file.write(f"{key} = {updates_dict[key]}\n")
                else:
                    file.write(line)
            else:
                file.write(line)

if __name__ == "__main__":
    download_config()
    download_blob()
    train_test_creation()
    obj_data_file = dir_path + "/data/1/obj.data"  # Path to your obj.data file
    updates_dict = {
        "classes": "1",
        "train": dir_path + "/data/1/train.txt",
        "valid":dir_path + "/data/1/val.txt",
        "names": dir_path + "/data/1/obj.names",
        "backup": dir_path + "/data/1/yolov4_tiny"
    }

    update_all_keys_in_obj_data(obj_data_file, updates_dict)
