import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.io import read_as_csv
from utils.pre_processing import read_image, label_to_index
import os.path
import joblib
from config  import  MODEL_CHECKPOINT_PATH   
from utils.load_config import config_laoder
from utils.pre_processing import image_transforms, label_transforms
from models.models import ModelZoo


def train(data_root, train_csv, test_csv, model, checkpoint_path):
# load csv
    train_path = os.path.join(data_root, train_csv)
    test_path = os.path.join(data_root, test_csv)
    train_files, train_labels = read_as_csv(train_path)
    test_files, test_labels = read_as_csv(test_path)

    # Apply the image_transforms function to train_files and test_files
    X_train = np.array(
        [image_transforms(file, label) for file, label in zip(train_files, train_labels)]
    )
    # print(X_train)
    Y_train = np.array([label_transforms(lab) for lab in train_labels])

    X_test = np.array(
        [image_transforms(file, label) for file, label in zip(test_files, test_labels)]
    )
    Y_test = np.array([label_transforms(lab) for lab in test_labels])


    

    clf = model
    clf.fit(X_train, Y_train)

    # pred=clf.predict(x_predict)
    # print("Pred:",[index_to_label[p]for p in pred])
    # dict_proba=[]
    # for proba in clf.predict_proba(x_predict):
    #     dict_proba.append({index_to_label[i]:p for i,p in enumerate(proba)})
    # print("pred prob:",dict_proba)

    print("Train score:", clf.score(X_train, Y_train))
    #print("Test score:", clf.score(X_test, Y_test))

    # Prediction
    #print("Before save:",clf.predict(X_test))

    # Save the model
    joblib.dump(clf, checkpoint_path)

    # Load the k-NN model from the file
    loaded_knn_model = joblib.load(checkpoint_path)


    # print(loaded_knn_model.predict(X_test))
    
if __name__ == "__main__":
    
    # parser = ArgumentParser(
    #     prog= "train", 
    #     description = "script to train the model"
    # )
    # parser.add_argument("-c", "--config")
    
    configs = config_laoder("configs\config.yaml")
    models = ModelZoo(**configs["model"]).get_model()
    #print(configs)
    train(data_root=configs["data_root"], train_csv=configs["train_csv"], test_csv=configs["test_csv"], model= models, checkpoint_path=configs["checkpoint_path"])