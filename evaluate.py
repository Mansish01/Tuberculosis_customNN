import torch
from models.customNN import FirstNeural, Model
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, cohen_kappa_score
from datasets.image_dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T     
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE =1
val_csv_path = r"data/test.csv"
total_labels = []
total_predictions = []
class_labels = ["Normal", "Tuberculosis"]

transforms= T.Compose([
        T.Resize((256, 256)), 
        T.ToTensor()])

val_dataset =ImageDataset( csv_path= val_csv_path , transforms= transforms)

   
val_data_loader= DataLoader(
        val_dataset, 
        batch_size = BATCH_SIZE,
        shuffle = True
    )

model_path = r"artifacts/run-2023-12-21-17-00-56/ckpt-Model-val=0.970-epoch=18"
checkpoint = torch.load(model_path)

model_state_dict  = checkpoint['model_state_dict']

model = Model(img_size= 256, num_channels=3, num_labels=2)
model.load_state_dict(model_state_dict)
model.eval()


for images , labels in val_data_loader:
           
            model_out = model(images)
           
            model_out = F.log_softmax(model_out , dim =1)
            preds = torch.argmax(model_out, dim=1)
            label= labels.numpy()
            prediction= preds.numpy()
        #     print(label , prediction)
            total_labels .append(label)
            total_predictions.append(prediction)
            

cm = confusion_matrix(total_labels , total_predictions)
              
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= class_labels) # display_labels=labels
disp.plot()
plt.show()
