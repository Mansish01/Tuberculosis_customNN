import torch
from torch.nn import Linear
from torch import nn
import torch.nn.functional as F
        
class Model(nn.Module):
    def __init__(self, img_size : int, num_channels:int, num_labels : int):
        
        super(Model, self).__init__()
        
        self.conv =nn.Conv2d(3,3, kernel_size=3 , padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_1= nn.Conv2d(3,1, kernel_size=3 , padding=1)
        self.Linear = nn.Linear(1*32*32 , 2)   


    def forward(self, x):
         x = self.conv(x)
         x= self.max_pool(x)
         
         x = self.conv(x)
         x = self.max_pool(x)
         
         x = self.conv_1(x)
         x= self.max_pool(x)

        # Flatten the tensor before applying the linear layer
         x = x.view(x.size(0), -1)
    
        # Linear layer
         x = self.Linear(x)

         return x

       
         
class SophisticatedModel(nn.Module):
    # def __init__(self, num_classes=2):
    def __init__(self, img_size : int, num_channels:int, num_labels : int):
        super(SophisticatedModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 16 * 16, 4096)
        self.relu7 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu8 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096, num_labels)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.maxpool2(x)
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.relu7(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu8(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x 
         
       

IMAGE_INPUT_CHANNEL = 'RGB'
KERNEL_SIZE = 3
POOL_KERNEL_SIZE = 2

class TuberculosisCNNReduced(nn.Module):
    def __init__(self):
        super(TuberculosisCNNReduced, self).__init__()
        # Define a simplified CNN architecture with 10 layers in total
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=KERNEL_SIZE, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=2),  # Layer 1

            nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=2),  # Layer 2

            nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=2),  # Layer 3

            nn.Conv2d(128, 256, kernel_size=KERNEL_SIZE, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=2),  # Layer 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * (256 // (2 ** 4)) * (256 // (2 ** 4)), 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Layer 5
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),  # Layer 6
            nn.Linear(256, 2),  # Layer 7
        )

    def forward(self, x):
        x = self.features(x)  # Pass input through feature extractor
        x = torch.flatten(x, 1)  # Flatten the features for the classifier
        x = self.classifier(x)  # Pass through the classifier
        return x  
                                                                              