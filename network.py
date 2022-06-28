import torch
import torch.nn as nn
from blocks import BasicBlock 


class Generator(nn.Module) : 
    def __init__(self, in_channels, out_channels) : 
        self.encoders = nn.ModuleList([
            BasicBlock(in_channels, 64, upsample = False, batchnorm = True), 
            BasicBlock(64, 128, upsample = False), 
            BasicBlock(128, 256, upsample = False), 
            BasicBlock(256, 512, upsample = False), 
            BasicBlock(512, 512, upsample = False), 
            BasicBlock(512, 512, upsample = False),
            BasicBlock(512, 512, upsample = False),
            BasicBlock(512, 512, upsample = False)
        ])

        self.decoders = nn.ModuleList([
            BasicBlock(512, 512, dropout = True), 
            BasicBlock(512 * 2, 512, dropout = True), 
            BasicBlock(512 * 2, 512, dropout = True),
            BasicBlock(512 * 2, 256, dropout = True), 
            BasicBlock(256 * 2, 128), 
            BasicBlock(128 * 2, 64) 
        ])

        self.decoder_final = nn.ConvTranspose2d(64 * 2, out_channels, \
            kernel_size = 4, stride = 2, padding= 1)
        
    
    def forward(self, x) :
        skips = [] 
        for encoder in self.encoders : 
            x = encoder(x)
            skips.append(x)
        skips = list(reversed(skips[:-1]))
        for decoder, skip in zip(self.decoders, skips) :
            x = torch.cat([x, skip], axis = 1)
            x = decoder(x)
        x = self.decoder_final(x)
        return nn.Tanh(x)



class Discriminator(nn.Module) : 
    def __init__(self, in_channels) :
        super().__init__()
        self.conv = nn.ModuleList([
            BasicBlock(in_channels, 64, batchnorm = False, upsample = False),
            BasicBlock(64, 128, upsample = False),
            BasicBlock(128, 256, upsample = False),
            BasicBlock(256, 512, upsample = False), 
            BasicBlock(512, 1, kernel_size = 1, upsample = False)
        ])

    def forward(self, x) : 
        for block in self.conv : 
            x = block(x)
        return x