import os
from os.path import join 
from utils.io import get_image_label_pairs , save_as_csv
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(prog="train test split",description= "input the value for train test split")

parser.add_argument("--test-size", type=float,default=0.2, help="input the test size for the model", required=True)
parser.add_argument("--random-state", type=int,default=42, help="value so that same random splitting can be done")
parser.add_argument("--Data-dir", type=str, help="input the required dicrectory")

args= parser.parse_args()



DATA_DIR= args.Data_dir
print(DATA_DIR)

data_folders= os.listdir(DATA_DIR)

x=[]
y=[]

for folder in data_folders:
    data_path= join(DATA_DIR, folder)
    label = folder
    
    files_names, label = get_image_label_pairs(data_path , label)
    x.extend(files_names)
    y.extend(label)
    
#print(len(x), len(y))

#train and test split 

#print(args.test_size)

#test_size = args.test_size
#random_state = args.random_state

x_train, x_test , y_train , y_test= train_test_split(x , y, test_size=args.test_size , random_state=args.random_state)
# print(len(x_train), len(x_test), len(y_train), len (y_test))

save_as_csv(x_train , y_train, 'data/train.csv')
save_as_csv(x_test , y_test , "data/test.csv")


