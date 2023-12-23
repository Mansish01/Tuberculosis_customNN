import joblib
import numpy as np
from config import MODEL_CHECKPOINT_PATH
from utils.pre_processing import image_transforms, label_to_index, index_to_label
from utils.io import read_as_csv
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
#from evaluate import display_grid
# Load the model
loaded_knn_model = joblib.load(MODEL_CHECKPOINT_PATH)


train_files, train_labels = read_as_csv("data/train.csv")
test_files, test_labels = read_as_csv("data/test.csv")


X_test = np.array(
    [image_transforms(file, label) for file, label in zip(test_files, test_labels)]
)
X_train = np.array(
    [image_transforms(file, label) for file, label in zip(train_files, train_labels)]
)
y_test = np.array([label_to_index(lab) for lab in test_labels])
neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(X_test, y_test)

indices_of_neighbors=loaded_knn_model.kneighbors(X_test[:4],n_neighbors=3,return_distance =False)
print(indices_of_neighbors)


# Make predictions using the KNN model on the test data
y_pred = loaded_knn_model.predict(X_test)
y_pred_labels=np.array([ index_to_label(p) for p in y_pred])
fig, axs = plt.subplots(4,4,figsize=(8, 8))

for i in range(4):

    axs[i, 0].imshow(X_test[i].reshape(256,256,3))
    axs[i, 0].set_title(f"Test Image:{train_labels[i]}\n Predicted Image:{y_pred_labels[i]}")
    axs[i, 0].axis('off')

    for j, idx in enumerate(indices_of_neighbors[i]):

        axs[i, j+1].imshow(X_train[idx].reshape(256,256,3))
        axs[i, j+1].set_title(f"Neighbor:{train_labels[idx]}")
        axs[i, j+1].axis('off')

plt.suptitle("Nearest Neighbors Images", fontsize=10)
plt.tight_layout()
plt.show()