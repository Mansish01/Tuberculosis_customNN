import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn 
from models.customNN import FirstNeural
from utils.io import read_as_csv
from utils.pre_processing import label_to_index, image_transforms
import numpy as np

IMG_SIZE = 256
NUM_CHANNELS= 3
NUM_LABELS = 4

#create model object 
model_nn = FirstNeural(img_size =IMG_SIZE, num_channels = NUM_CHANNELS, num_labels= NUM_LABELS)

#create dataset
