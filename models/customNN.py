import torch 
from torch.nn.functional import relu
from torch.nn import Linear
from torch import nn

        
class Model(nn.Module):
    def __init__(self, img_size : int, num_channels:int, num_labels : int):
        super().__init__()
        
        # self.Linear_1= nn.Linear(5, 10)
        # self.Linear_2 = nn.Linear(10 ,2)
        # self.sigmoid  = nn.Sigmoid()
        self.conv =nn.Conv2d(3,3, kernel_size=3 , padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_1= nn.Conv2d(3,1, kernel_size=3 , padding=1)
        
        self.Linear = nn.Linear(1*32*32 , 2)        
    
    def forward(self , x):
         

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
                                                                                