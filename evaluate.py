import torch
from models.customNN import Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, cohen_kappa_score
from datasets.image_dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T     
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 1
val_csv_path = r"data/test.csv"
total_labels = []
total_predictions = []
class_labels = ["Normal", "Tuberculosis"]

transforms = T.Compose([
    T.Resize((256, 256)), 
    T.ToTensor()
])

test_dataset = ImageDataset(csv_path=val_csv_path, transforms=transforms)

test_data_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True
)

model_path = r"artifacts/run-2024-02-24-22-28-53/ckpt-Model-val=0.974-epoch=24"
checkpoint = torch.load(model_path)

model_state_dict = checkpoint['model_state_dict']

model = Model(img_size=256, num_channels=3, num_labels=2)
model.load_state_dict(model_state_dict)
model.eval()

for images, labels in test_data_loader:
    model_out = model(images)
    model_out = F.log_softmax(model_out, dim=1)
    preds = torch.argmax(model_out, dim=1)
    label = labels.numpy()
    prediction = preds.numpy()
    total_labels.append(label)
    total_predictions.append(prediction)

# Calculate and print the classification report
classification_rep = classification_report(np.concatenate(total_labels), np.concatenate(total_predictions), target_names=class_labels)
print("Classification Report:\n", classification_rep)

# Plot the confusion matrix
cm = confusion_matrix(np.concatenate(total_labels), np.concatenate(total_predictions))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot()
plt.show()
