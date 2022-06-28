from torchvision import transforms
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from utils.get_data import CustomDataset
from model import CycleGAN
import pytorch_lightning as pl 

if __name__ == "__main__" : 


    root = "/Users/kunwoona/Desktop/Projects/Datasets/face2anime"       # write your own directory here! 
    lambda_recon = 100  
    n_epochs =40 
    display_step = 2
    batch_size = 4
    lr = 0.0003
    target_size = 256 
    device = "mps"

    dataset= CustomDataset(root, mode= "train", unaligned=True)
    dataloader = DataLoader(dataset, batch_size, shuffle = True, num_workers=8)

    cycleGAN = CycleGAN(3, 3, learning_rate= lr, \
        lambda_ = lambda_recon, display_step=display_step)
    trainer = pl.Trainer(max_epochs = 40, accelerator = "mps", devices = 1)

    trainer.fit(cycleGAN, dataloader)