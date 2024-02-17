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

        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.conv7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.conv8 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.conv10 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear(128 * 16 * 16, 128)  # Adjusted size due to max pooling
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        # self.fc4 = nn.Linear(128, 128)
        # self.fc5 = nn.Linear(128, 128)
        # self.fc6 = nn.Linear(128, 128)
        # self.fc7 = nn.Linear(128, 2)

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

        # x = torch.relu(self.conv1(x))
        # x = torch.relu(self.conv2(x))
        # x = torch.relu(self.conv3(x))
        # x = torch.relu(self.conv4(x))
        # x = self.pool(x)
        # x = torch.relu(self.conv5(x))
        # x = torch.relu(self.conv6(x))
        # x = torch.relu(self.conv7(x))
        # x = self.pool(x)
        # x = torch.relu(self.conv8(x))
        # x = torch.relu(self.conv9(x))
        # x = self.pool(x)
        # x = torch.relu(self.conv10(x))
        # x = torch.relu(self.conv11(x))
        # x = self.pool(x)
        # x = x.view(-1, 128 * 16 * 16)  # Adjusted size due to max pooling
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        # x = torch.relu(self.fc5(x))
        # x = torch.relu(self.fc6(x))
        # x = self.fc7(x)
        # return torch.softmax(x, dim=1)
        
      










        # self.conv =nn.Conv2d(3,3, kernel_size=3 , padding=1)
        # self.max_pool = nn.MaxPool2d(kernel_size=2)
        # self.conv_1= nn.Conv2d(3,1, kernel_size=3 , padding=1)
        
        # self.Linear = nn.Linear(1*32*32 , 2)   

       
         

       
         
       

                                                                                