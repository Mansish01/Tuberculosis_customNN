import torch
from torch.nn import Linear
from torch import nn
import torch.nn.functional as F
        
class Model(nn.Module):
    def __init__(self, img_size : int, num_channels:int, num_labels : int):
        super().__init__()
        
        # self.conv =nn.Conv2d(3,3, kernel_size=3 , padding=1)
        # self.max_pool = nn.MaxPool2d(kernel_size=2)
        # self.conv_1= nn.Conv2d(3,1, kernel_size=3 , padding=1)
        
        # self.Linear = nn.Linear(1*32*32 , 2)   

         # Block 1
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification block
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * (img_size // 32) * (img_size // 32), 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,  num_labels)     
    
    def forward(self , x):
         

        #  x = self.conv(x)
        #  x= self.max_pool(x)
        #  x = self.conv(x)
        #  x = self.max_pool(x)
        #  x = self.conv_1(x)
        #  x= self.max_pool(x)

        # # Flatten the tensor before applying the linear layer
        #  x = x.view(x.size(0), -1)
    
        # # Linear layer
        #  x = self.Linear(x)
         
          # Block 1
        x = F.relu(self.block1_conv1(x))
        x = F.relu(self.block1_conv2(x))
        x = self.block1_pool(x)

        # Block 2
        x = F.relu(self.block2_conv1(x))
        x = F.relu(self.block2_conv2(x))
        x = self.block2_pool(x)

        # Block 3
        x = F.relu(self.block3_conv1(x))
        x = F.relu(self.block3_conv2(x))
        x = F.relu(self.block3_conv3(x))
        x = self.block3_pool(x)

        # Block 4
        x = F.relu(self.block4_conv1(x))
        x = F.relu(self.block4_conv2(x))
        x = F.relu(self.block4_conv3(x))
        x = self.block4_pool(x)

        # Block 5
        x = F.relu(self.block5_conv1(x))
        x = F.relu(self.block5_conv2(x))
        x = F.relu(self.block5_conv3(x))
        x = self.block5_pool(x)

        # Classification block
        # x = self.flatten(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)

                                                                                