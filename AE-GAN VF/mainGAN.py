from __future__ import print_function, division
#import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from AutoEncoder import Encoder, Decoder
from Discriminator import Discriminator
from Generator import Generator
from Variables import *
from Dataset import *
from getData import get_velocity_field
from Norm import *
from nadam import Nadam


#########################
# Instantiating Dataset #
#########################
tracer_dataset = TracerDataset(transform = ToTensor())
tracer_latent_dataset = TracerLatentDataset(transform = ToTensor())

# Decide whihch device we want to run on
if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


#########################
# Weight Initialisation #
#########################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ###################################
# # Instantiating Encoder from file #
# ###################################
# netEnc = Encoder(ngpu).to(device)
# print(netEnc)

# # Handle multi-gpu if required
# if (device.type == 'cuda') and (ngpu > 1):
#     netEnc = nn.DataParallel(netEnc, list(range(ngpu)))

# # Apply the weights initialiser function to randomly initalise all weights
# # to mean=0 and sd=0.2
# netEnc.apply(weights_init)

###############################
# Instantiating Discriminator #
###############################
netD = Discriminator(ngpu).to(device)
print(netD)

# Handle multi-gpu if required
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights initialiser function to randomly initalise all weights
# to mean=0 and sd=0.2
netD.apply(weights_init)

###########################
# Instantiating Generator #
###########################
netG = Generator(ngpu).to(device)
print(netG)

# Handle multi-gpu if required
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights initialiser function to randomly initalise all weights
# to mean=0 and sd=0.2
netG.apply(weights_init)

#################################
# Loss Functions and Optimisers #
#################################

## Loss Functions
mse_loss = nn.MSELoss()
#bce_loss = nn.BCELoss()
bce_loss = nn.BCEWithLogitsLoss()

# *** TODO: Physics based Loss *** #

# Setup Adam optimizers
# optimizerAE = optim.Adam(netAE.parameters(), lr=lr, betas=(beta1, 0.999)) # Nadam optim?
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.SGD(netD.parameters(), lr=lr)
#optimizerEnc = Nadam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999)) 
optimizerG = Nadam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerD = Nadam(netD.parameters(), lr=lr, betas=(beta1, 0.999))


###################
# Loading Encoder #
###################
# checkpoint = torch.load("/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/AutoEncoder64")
# netEnc.load_state_dict(checkpoint['netEnc_state_dict'])
# optimizerEnc.load_state_dict(checkpoint['optimizerEnc_state_dict'])

############
# Training #
############
epoch_list = []
g_loss_list = []
mse_loss_list = []
bce_loss_list = []
d_loss_list = []
val_loss_list = []

# Splitting data between train and validation sets
ints = list(range(time_steps))
random.shuffle(ints)
int_to_split = int(val_percent * time_steps)
train_ints = ints[int_to_split:]
val_ints = ints[:int_to_split]

for epoch in range(num_epochs_GAN):
    # Getting batches for Training set
    batch_indicies = list(BatchSampler(RandomSampler(train_ints), batch_size=batch_size, drop_last=True)) #Should include workers?
    batch_indicies_incr = [[i + 1 for i in item] for item in batch_indicies]

    dataloader = DataLoader(tracer_dataset, batch_sampler=batch_indicies, num_workers=2) # Should add workers
    dataloader_incr = DataLoader(tracer_dataset, batch_sampler=batch_indicies_incr, num_workers=2)
    dataloader_latent = DataLoader(tracer_latent_dataset, batch_sampler=batch_indicies, num_workers=2) # Should add workers

    #print("About to enter loops")
    for i_batch, (sample_batched, sample_batched_incr, sample_batched_latent) in enumerate(zip(dataloader, dataloader_incr, dataloader_latent)):
        #print(i_batch)        
        data = sample_batched.to(device=device, dtype=torch.float)
        data_incr = sample_batched.to(device=device, dtype=torch.float)
        data_latent = sample_batched_latent.to(device=device, dtype=torch.float)
        
        #################################################################
        # (1) Update Discriminator: maximise log(D(x)) + log(1-D(G(z))) #
        #################################################################

        ######################################
        ## Training with the all-real batch ##
        ######################################
        netD.zero_grad()
        real_label = random.randint(70,100)/100
        label = torch.full((batch_size,), real_label, device=device, dtype = torch.float)
        outputD_real = netD(data)
        label = label.unsqueeze(1).unsqueeze(1)
        errD_real = bce_loss(outputD_real, label) 

        ######################################
        ## Training with the all-fake batch ##
        ######################################

        #outputEnc = netEnc(data)
        #outputG = netG(outputEnc.detach())
        outputG = netG(data_latent)
        outputD_fake = netD(outputG.detach()) 
        
        fake_label = random.randint(0,30)/100
        label = torch.full((batch_size,), fake_label, device=device, dtype = torch.float)
        # Calculate loss
        label = label.unsqueeze(1).unsqueeze(1)
        errD_fake = bce_loss(outputD_fake, label)


        # Sum gradients and update
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        ##################################################################################
        # (2) Update Generator/AE: maximise log(D(G(z))) AND minimise MSE + Physics Loss #
        ##################################################################################

        #netEnc.zero_grad() # Just to avoid wasting memory
        netG.zero_grad()
        label.fill_(real_label)

        # Since we just updated the Discriminator, we perform
        # a forward pass through discrim again
        output = netD(outputG) # outputG does not contain any Encoder graphs
        # Calculate Generator's loss
        errBCE = bce_loss(output, label)
        errMSE = mse_loss(outputG, data_incr)

        errG = errBCE + alpha*errMSE # + PBL

        # Calculate Gradients
        errG.backward() 
        # Update Gradients
        optimizerG.step()

        # Display losses
        if i_batch % 50 == 0:
            print("Training Losses")
            print("Epoch: ", epoch, " | i: ", i_batch)
            print("Discriminator Loss: ", errD.item())
            print("Generator Loss: ", errG.item())
            print("Gen BCE Loss:", errBCE.item())
            print("Gen MSE Loss:", alpha*errMSE.item())
        
    # Validation errors
    ## Getting batches for Validation set    
    val_indicies = list(BatchSampler(RandomSampler(val_ints), batch_size=batch_size, drop_last=True)) #Should include workers?
    val_indicies_incr = [[i + 1 for i in item] for item in val_indicies]

    val_dataloader = DataLoader(tracer_dataset, batch_sampler=val_indicies, num_workers=2) # Should add workers
    val_dataloader_incr = DataLoader(tracer_dataset, batch_sampler=val_indicies_incr, num_workers=2)
    val_dataloader_latent = DataLoader(tracer_latent_dataset, batch_sampler=val_indicies, num_workers=2) # Should add workers
    
    ## Pass through networks and calculate losses
    errG_val_mse = 0
    for i_batch, (sample_batched_incr, sample_batched_latent) in enumerate(zip(val_dataloader_incr, val_dataloader_latent)):
        data_incr = sample_batched.to(device=device, dtype=torch.float)
        data_latent = sample_batched_latent.to(device=device, dtype=torch.float)

        # Pass through Generator
        netG.zero_grad()
        outputG = netG(data_latent).detach()
        errG_val_mse += mse_loss(outputG, data_incr)
    errG_val_mse /= len(val_dataloader)
    errG_val_mse *= alpha
    print("Validation G_MSE Loss: ", errG_val_mse.item())

    # Storing losses per epoch to plot
    epoch_list.append(epoch)
    g_loss_list.append(errG.item())
    bce_loss_list.append(errBCE.item())
    mse_loss_list.append(alpha*errMSE.item())
    d_loss_list.append(errD.item())
    val_loss_list.append(errG_val_mse.item())

    
# Save graph info   
    
plt.plot(epoch_list, g_loss_list, label="Generator")
plt.plot(epoch_list, d_loss_list, label="Discriminator")
plt.plot(epoch_list, bce_loss_list, label="G_BCE Loss")
plt.plot(epoch_list, mse_loss_list, label="G_MSE Loss")
plt.plot(epoch_list, val_loss_list, label="Validation G_MSE")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Training complete. Saving model...")
torch.save({
            'netG_state_dict': netG.state_dict(), 
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict() },
            "/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/GANVF")
print("Model has saved successfully!")
