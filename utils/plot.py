import matplotlib.pyplot as plt
import torch, torchvision
import torch.nn as nn 
import os 

def display_progress(real_image, fake_image, figsize = (12, 6)) : 
    real_image = real_image.detach().cpu().permute(1, 2, 0)
    fake_image = fake_image.detach().cpu().permute(1, 2, 0)
    fig, ax = plt.subplot(1, 2, figsize = figsize)
    ax[0].imshow(real_image)
    ax[1].imshow(fake_image)
    #ax[0].title.set_text('Input image')
    #ax[1].title.set_text('Result image')
    #plt.axis('off')
    plt.show()
    # plt.savefig(path)



def save_image(image, epoch, XtoY : bool) : 
    if not(os.path.isdir('/ImageOutput')):
        os.mkdir('/ImageOutput')
    if XtoY :
        torchvision.utils.save_image(image, f"/ImageOutput/XtoY{epoch}.png") 
    else : 
        torchvision.utils.save_image(image, f"/ImageOutput/YtoX{epoch}.png")