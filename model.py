import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import pytorch_lightning as pl 

from buffer import ImageBuffer
from cycleGAN.utils.plot import save_image
from network import Generator, Discriminator
from utils.get_data import CustomDataset
from utils.plot import display_progress
# from utils.weight import initiate_weight 


###################################################
## TODO : apply 'initiate_weight' functionality! ##
###################################################

class CycleGAN(pl.LightningModule) :

    def __init__(self, in_channels, out_channels, learning_rate, display_step = 4, lambda_ = 150) :
        super().__init__()
        self.save_hyperparameters()
        self.display_step = display_step
        self.lambda_ = lambda_ 
        self.lr = learning_rate 
        self.G = Generator(in_channels, out_channels)
        self.F = Generator(out_channels, in_channels)
        self.D_X = Discriminator(in_channels)
        self.D_Y = Discriminator(out_channels)

        self.loss_GAN = nn.BCEWithLogitsLoss()
        self.loss_cyc = nn.L1Loss()

        self.fake_X_buffer = ImageBuffer(50)
        self.fake_Y_buffer = ImageBuffer(50)
        self.mode = "XtoY"      # default mode.  ** TODO : implement the functionality to switch the mode. ** 


    def forward(self, image_X, image_Y) : 
        self.fake_Y = self.G(image_X)
        self.recon_X = self.F(self.fake_Y)
        self.fake_X = self.F(image_Y)
        self.recon_Y = self.G(self.fake_X)
    
    ##############################################
    ### TODO : take use of this functionality! ###
    ##############################################

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Paramete    rs:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        XtoY = self.opt.direction == 'AtoB'
        self.real_X = input['X' if XtoY else 'Y'].to(self.device)
        self.real_Y = input['Y' if XtoY else 'X'].to(self.device)
        # self.image_paths = input['A_paths' if XtoY else 'B_paths']
    

    
    def _G_step(self, image_X, image_Y) :
        fake_images = self.G(image_X)
        D_Y_logits = self.D_Y(fake_images)
        adversarial_loss = self.loss_GAN(D_Y_logits, torch.zeros_like(D_Y_logits))
        cyclic_loss = self.loss_cyc(fake_images, image_Y)
        return adversarial_loss + self.lambda_ * cyclic_loss
    

    def _F_step(self, image_X, image_Y) : 
        fake_images = self.F(image_Y)
        D_X_logits = self.D_X(fake_images)
        adversarial_loss = self.loss_GAN(D_X_logits, torch.zeros_like(D_X_logits))
        cyclic_loss = self.loss_cyc(fake_images, image_X)
        return adversarial_loss + self.lambda_ * cyclic_loss


    def _D_X_step(self, image_X) : 
        #fake_images = self.F(image_Y)
        fake_images = self.fake_X_buffer.query(self.fake_X)
        fake_logits = self.D_X(fake_images)
        real_logits = self.D_X(image_X)
        fake_loss = self.loss_GAN(fake_logits, torch.ones_like(fake_logits))
        real_loss = self.loss_GAN(real_logits, torch.zeros_like(real_logits))
        return (fake_loss + real_loss) / 2
    

    def _D_Y_step(self, image_Y) :
        fake_images = self.fake_Y_buffer.query(self.fake_Y)
        fake_logits = self.D_Y(fake_images)
        real_logits = self.D_Y(image_Y)
        fake_loss = self.loss_GAN(fake_logits, torch.ones_like(fake_logits))
        real_loss = self.loss_GAN(real_logits, torch.zeros_like(real_logits))
        return (fake_loss + real_loss) / 2


    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        G_opt = torch.optim.Adam(self.G.parameters(), lr = lr)
        F_opt = torch.optim.Adam(self.F.parameters(), lr = lr)
        D_X_opt = torch.optim.Adam(self.D_X.parameters(), lr = lr)
        D_Y_opt = torch.optim.Adam(self.D_Y.parameters(), lr = lr) 
        return G_opt, F_opt, D_X_opt, D_Y_opt


    def training_step(self, batch, batch_idx, optimizer_idx) :
        image_X, image_Y = batch 
        self.forward(image_X, image_Y)
        loss = None 
        if optimizer_idx == 0 : 
            loss = self._G_step(image_X, image_Y)
            self.log('Generator G Loss', loss)
        elif optimizer_idx == 1 : 
            loss = self._F_step(image_X, image_Y)
            self.log('Generrator F loss', loss)
        elif optimizer_idx == 2 : 
            loss = self._D_X_step(image_X) 
            self.log('Discriminator X loss', loss)
        elif optimizer_idx == 3 : 
            loss = self._D_Y_step(image_Y)
            self.log('Discriminator Y loss', loss)
        
        if self.current_epoch % self.display_step == 0 and batch_idx == 0 and optimizer_idx ==0 : 
            fake_image = self.G(image_X).detach()
            display_progress(image_X[0], fake_image[0])
        
        ############  Save the result!  ############
        fake_X = self.F(image_Y).detach()
        fake_Y = self.G(image_X).detach()
        save_image(fake_X, self.current_epoch, False)
        save_image(fake_Y, self.current_epoch, True)

        return loss 