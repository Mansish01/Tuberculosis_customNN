from models.customNN import FirstNeural
import torch
nn= FirstNeural(img_size=512, num_labels=10)


BATCH_SIZE = 4
img = torch.zeros(size =(BATCH_SIZE, 512,512))
output = nn.forward(img)

assert  output.shape == (4, 10)
#print(output.shape)