import os
from os.path import join
from sklearn.model_selection import train_test_split
import argparse
import csv

def list_files(dir: str, file_extension: str) -> list:
    """given a directory, list all files with given extension"""
    if not os.path.isdir(dir):
        return None
    files = os.listdir(dir)
    return files

def get_image_label_pairs(dir: str, label: str) -> tuple:
    filenames = list_files(dir, "")
    labels = [label] * len(filenames)
    return filenames, labels

def save_as_csv(img_path: str, label: str, outfile):
    """save image path and save as csv file"""
    with open(outfile, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        for image_path, label in zip(img_path, label):
            writer.writerow([image_path, label])

def read_as_csv(csv_file):
    image_path = []
    labels = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_path.append(row[0])
            labels.append(row[1])
    return image_path, labels

parser = argparse.ArgumentParser(prog="train test split", description="input the value for train, validation, and test split")

parser.add_argument("--test-size", type=float, default=0.1, help="input the test size for the model")
parser.add_argument("--validation-size", type=float, default=0.1, help="input the validation size for the model")
parser.add_argument("--random-state", type=int, default=42, help="value so that the same random splitting can be done")
parser.add_argument("--Data-dir", type=str, help="input the required directory")

args = parser.parse_args()

DATA_DIR = args.Data_dir
print(DATA_DIR)

data_folders = os.listdir(DATA_DIR)

x = []
y = []

for folder in data_folders:
    data_path = join(DATA_DIR, folder)
    label = folder

    files_names, label = get_image_label_pairs(data_path, label)
    x.extend(files_names)
    y.extend(label)

# First, split into training and temp sets
x_train_temp, x_test, y_train_temp, y_test = train_test_split(x, y, test_size=args.test_size, random_state=args.random_state)

# Then, split the temp set into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=args.validation_size, random_state=args.random_state)

save_as_csv(x_train, y_train, 'data/train.csv')
save_as_csv(x_val, y_val, 'data/validation.csv')
save_as_csv(x_test, y_test, 'data/test.csv')
