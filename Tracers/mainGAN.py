import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision import transforms, utils
from AutoEncoder import *
from Discriminator import Discriminator
from Generator import Generator
from Variables import *
from Dataset import *
from getData import get_tracer
from Norm import *
from nadam import Nadam
import csv

#########################
# Instantiating Dataset #
#########################
tracer_dataset = TracerDataset(transform = ToTensor())

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

###################################
# Instantiating Encoder from file #
###################################
netEnc = Encoder(ngpu).to(device)

###############################
# Instantiating Discriminator #
###############################
netD = Discriminator(ngpu).to(device)
print(netD)

###########################
# Instantiating Generator #
###########################
netG = Generator(ngpu).to(device)
print(netG)

# Handle multi-gpu if required
if (device.type == 'cuda') and (ngpu > 1):
    netEnc = nn.DataParallel(netEnc, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights initialiser function to randomly initalise all weights
# to mean=0 and sd=0.2
netEnc.apply(weights_init)
netD.apply(weights_init)
netG.apply(weights_init)

#################################
# Loss Functions and Optimisers #
#################################

## Loss Functions
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

optimizerD = optim.SGD(netD.parameters(), lr=lr)
optimizerEnc = optim.Adam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999)) 
optimizerG = Nadam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

###################
# Loading Encoder # 
###################
# change directoy as needed
checkpoint = torch.load("/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/tAE") 
netEnc.load_state_dict(checkpoint['netEnc_state_dict'])
optimizerEnc.load_state_dict(checkpoint['optimizerEnc_state_dict'])

############
# Training #
############
epoch_list = []
g_loss_list = []
mse_loss_list = []
bce_loss_list = []
d_loss_list = []
val_loss_list = []

for epoch in range(num_epochs_GAN):
    # Getting batches for Training set
    batch_indicies = list(BatchSampler(RandomSampler(train_ints), batch_size=batch_size, drop_last=True))
    batch_indicies_incr = [[i + 1 for i in item] for item in batch_indicies]
    dataloader = DataLoader(tracer_dataset, batch_sampler=batch_indicies, num_workers=2)
    dataloader_incr = DataLoader(tracer_dataset, batch_sampler=batch_indicies_incr, num_workers=2)

    for i_batch, (sample_batched, sample_batched_incr) in enumerate(zip(dataloader, dataloader_incr)):
        data = sample_batched.to(device=device, dtype=torch.float)
        data_incr = sample_batched.to(device=device, dtype=torch.float)
        
        ##########################
        # Discriminator Training #
        ##########################

        ######################################
        ## Training with the all-real batch ##
        ######################################
        netD.zero_grad()
        real_label = random.randint(70,100)/100
        label = torch.full((batch_size,), real_label, device=device, dtype = torch.float)
        label = label.unsqueeze(1).unsqueeze(1)

        outputD_real = netD(data)
        errD_real = bce_loss(outputD_real, label) 

        ######################################
        ## Training with the all-fake batch ##
        ######################################
        fake_label = random.randint(0,30)/100
        label = torch.full((batch_size,), fake_label, device=device, dtype = torch.float)
        label = label.unsqueeze(1).unsqueeze(1)
    
        with torch.no_grad(): # Do not store graphs for Encoder
            outputEnc = netEnc(data).detach()
        outputG = netG(outputEnc)
        outputD_fake = netD(outputG.detach()) 
        errD_fake = bce_loss(outputD_fake, label)

        # Sum Errors, Calculate Gradients and Update
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        ######################
        # Generator Training #
        ######################
        netEnc.zero_grad()
        netG.zero_grad()
        label.fill_(real_label)

        # Since we just updated the Discriminator, we perform
        # a forward pass through D again
        output = netD(outputG) # outputG does not contain any Encoder graphs
        errBCE = bce_loss(output, label)
        errMSE = mse_loss(outputG, data_incr)

        errG = errBCE + alpha*errMSE 

        # Calculate Gradients and Update
        errG.backward() 
        optimizerG.step()

        #Display losses
        if i_batch % 50 == 0:
            print("Training Losses")
            print("Epoch: ", epoch, " | i: ", i_batch)
            print("Discriminator Loss: ", errD.item())
            print("Generator Loss: ", errG.item())
            print("Gen BCE Loss:", errBCE.item())
            print("Gen MSE Loss:", alpha*errMSE.item())
        
    ## Pass through networks and calculate losses
    errG_val_mse = 0
    for i in val_ints:
        val_tracer = get_tracer(i)
        val_tracer = torch.from_numpy(val_tracer).unsqueeze(1).to(device=device, dtype=torch.float)
        val_tracer_incr = get_tracer(i+1)
        val_tracer_incr = torch.from_numpy(val_tracer_incr).unsqueeze(1).to(device=device, dtype=torch.float)

        # Pass through Encoder + Generator
        val_outputEnc = netEnc(val_tracer).detach()
        val_outputG = netG(val_outputEnc).detach()

        errG_val_mse += mse_loss(val_outputG, val_tracer_incr)

    
    ## Display Validation losses
    errG_val_mse /= len(val_ints)
    errG_val_mse *= alpha
    print("Validation G_MSE Loss: ", errG_val_mse.item())
    
    # Storing losses per epoch to plot
    epoch_list.append(epoch)
    g_loss_list.append(errG.item())
    bce_loss_list.append(errBCE.item())
    mse_loss_list.append(alpha*errMSE.item())
    d_loss_list.append(errD.item())
    val_loss_list.append(errG_val_mse.item())

    if epoch % 200 == 0:
        print("Saving model at epoch ", epoch)
        torch.save({
                    'netG_state_dict': netG.state_dict(), 
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict() },
                    "/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/tGAN" + str(epoch))
        print("Model has saved successfully!")

        # Save graph outputs
        newfilePath = '/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/GANgraphs/tGAN' + str(epoch) + '.csv'
        rows = zip(epoch_list, g_loss_list, d_loss_list, bce_loss_list, mse_loss_list, val_loss_list)
        with open(newfilePath, "w") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

    
print("Training complete. Saving model...")
torch.save({
            'netG_state_dict': netG.state_dict(), 
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict() },
            "/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/tGAN1000")
print("Model has saved successfully!")

# Save graph outputs
newfilePath = '/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/GANgraphs/tGAN1000.csv'
rows = zip(epoch_list, g_loss_list, d_loss_list, bce_loss_list, mse_loss_list, val_loss_list)
with open(newfilePath, "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

plt.plot(epoch_list, g_loss_list, label="Generator")
plt.plot(epoch_list, d_loss_list, label="Discriminator")
plt.plot(epoch_list, bce_loss_list, label="G_BCE Loss")
plt.plot(epoch_list, mse_loss_list, label="G_MSE Loss")
plt.plot(epoch_list, val_loss_list, label="Validation G_MSE")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
