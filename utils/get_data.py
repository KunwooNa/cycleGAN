import os
import glob
from pathlib import Path
from PIL import Image
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random 


'''
- Implementation of Custom dataset. 
    * TODO : implement functionalities that can enable flip, crop oper. 
    * Note : for cycleGAN, we do not provide the model with {(real image, conditional image)}. 
    * Note : instead, we only give real images in each domain. 
    * Code reference : https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/datasets.py
'''



class CustomDataset(Dataset) : 

    def __init__(self, root, transforms_ = None, mode = "train", unaligned = True) :
        super().__init__()
        self.transforms = transforms.functional.to_tensor
        self.mode = mode
        self.unaligned = unaligned 
        self.files_X = sorted(glob.glob(os.path.join(root, mode, ('%sA' % mode) + '/*.*')))
        self.files_Y = sorted(glob.glob(os.path.join(root, mode, ('%sB' % mode) + '/*.*')))
    
    def __len__(self) : 
        return max(len(self.files_X), len(self.files_Y))


    def __getitem__(self, idx):
        image_X = self.transforms(Image.open(self.files_X[idx % len(self.files_X)]))
        
        if self.unaligned:
            image_Y = self.transforms(Image.open(self.files_Y[random.randint(0, len(self.files_Y) - 1)]))
        else:
            image_Y = self.transforms(Image.open(self.files_Y[idx % len(self.files_Y)]))

        return image_X, image_Y
