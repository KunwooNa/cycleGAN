import torch 
import torch.nn as nn 


###################################################
### TODO : more sophisticated version required. ###
###################################################

def initiate_weight(module) : 
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        torch.nn.init.normal_(module.weight, 0.0, 0.02)
    return None 