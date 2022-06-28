import os
from glob import glob
from pathlib import Path
import itertools 
from PIL import Image
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


'''
- Implementation of Custom dataset. 
    * TODO : implement functionalities that can enable flip, crop oper. 
    * Note : for cycleGAN, we do not provide the model with {(real image, conditional image)}. 
    * Note : instead, we only give real images in each domain. 
'''

class CustomDataset(Dataset) : 

    def __init__(self, path, size) :
        super().__init__()
        self.filenames = glob(str(Path(path) / "*"))
        self.size = size
    
    def __len__(self) : 
        return len(self.filenames)


    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(filename)
        image = transforms.functional.to_tensor(image)
        image_width = image.shape[2]

        real = image[:, :, : image_width // 2]
        condition = image[:, :, image_width // 2 :]

        target_size = self.size
        if target_size:
            condition = nn.functional.interpolate(condition, size=target_size)
            real = nn.functional.interpolate(real, size=target_size)

        return real, condition