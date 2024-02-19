import torch
from torch.utils.data import Dataset

from torch import nn 
# from models.customNN import FirstNeural
from io1 import read_as_csv
from util.pre_processing import label_to_index
import numpy as np
from torch import nn

# from models.customNN import FirstNeural
from io1 import read_as_csv
from util.pre_processing import label_to_index
import numpy as np
from os.path import join
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        images, labels = read_as_csv(csv_path)
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return f"<ImageDatset with {self.__len__()} samples>"

    def __getitem__(self, index):
        image_name = self.images[index]
        label_name = self.labels[index]
        image_path = join("data", "tuber_dataset", label_name, image_name)
        # print(image_path)
        image = Image.open(image_path).convert("RGB")

        label = label_to_index(label_name)
        if self.transforms:
            image = self.transforms(image)

        return image, label
