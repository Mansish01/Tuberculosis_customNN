import csv
import glob
import os

import pandas as pd

BASE_DIR=
# BASE_DIR="..\data
TRAIN_DIR=
TEST_DIR=


train_images=glob.glob(f"{TRAIN_DIR}/**/*.jpg")
test_images=glob.glob(f"{TEST_DIR}/**/*.jpg")

train_labels=[image_path.split('\\')[3]for image_path in train_images]
test_labels= [image_path.split('\\')[3]for image_path in test_images]

train_data = pd.DataFrame({'image_path': train_images, 'labels': train_labels})
test_data  = pd.DataFrame({'image_path': test_images, 'labels': test_labels})

# Save dataframes to CSV files
train_data.to_csv(os.path.join(BASE_DIR, 'train.csv'), index=False)
test_data.to_csv(os.path.join(BASE_DIR, 'test.csv'), index=False)














