import torch 
import torch.nn as nn 
from abc import ABC


class BasicModule(nn.Module, ABC) : 
    def __init__(self, in_channels, out_channels, upsample, \
        kernel_size = 4, stride = 2, padding = 1, activation = True, batchnorm = True, dropout = False, ) : 
        super().__init__()
        self.activation = activation 
        self.batchnorm = batchnorm 
        self.dropout = dropout 
        self.upsample = upsample 
        if upsample == True : 
            self.conv = nn.Conv2d(in_channels, out_channels, \
                 kernel_size, stride, padding)
        else : 
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, \
                kernel_size, stride, padding)
        
        if batchnorm : 
            self.bn = nn.BatchNorm2d(out_channels)
        if activation : 
            self.act = nn.LeakyReLU(0.2)
        if dropout : 
            self.drop = nn.Dropout2d(0.5)
        
    
    def forward(self, x) :
        x_ = self.conv(x)
        if self.batchnorm : 
            x_ = self.bn(x_)
        if self.activation : 
            x_ = self.act(x_)
        if self.dropout : 
            x_ = self.drop(x_)
        return x_ 
