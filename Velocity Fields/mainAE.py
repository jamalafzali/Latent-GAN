#from __future__ import print_function, division
#%matplotlib inline
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from AutoEncoder import Encoder, Decoder
from Variables import *
from Dataset import *
from getData import get_velocity_field
from Norm import *
from nadam import Nadam
import csv

#########################
# Instantiating Dataset #
#########################
velocity_dataset = VelocityFieldDataset(transform = ToTensor())

# Decide which device we want to run on
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

#############################
# Instantiating AutoEncoder #
#############################
netEnc = Encoder(ngpu).to(device)
netDec = Decoder(ngpu).to(device)
print(netEnc)
print(netDec)

# Handle multi-gpu if required
if (device.type == 'cuda') and (ngpu > 1):
    netEnc = nn.DataParallel(netEnc, list(range(ngpu)))
    netDec = nn.DataParallel(netDec, list(range(ngpu)))

# Apply the weights initialiser function to randomly initalise all weights
# to mean=0 and sd=0.2
netEnc.apply(weights_init)
netDec.apply(weights_init)

#################################
# Loss Functions and Optimisers #
#################################

## Loss Functions
mse_loss = nn.MSELoss()

optimizerEnc = Nadam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDec = Nadam(netDec.parameters(), lr=lr, betas=(beta1, 0.999))

############
# Training #
############
epoch_list = []
loss_list = []
val_loss_list = []

for epoch in range(num_epochs_AE):
    # Getting batches for Training set
    batch_indicies = list(BatchSampler(RandomSampler(train_ints), batch_size=batch_size, drop_last=True)) 
    dataloader = DataLoader(velocity_dataset, batch_sampler=batch_indicies, num_workers=2) 

    for i_batch, sample_batched in enumerate(dataloader):
        data = sample_batched.to(device=device, dtype=torch.float)

        netEnc.zero_grad()
        netDec.zero_grad()

        output = netEnc(data)
        output = netDec(output)

        errAE = mse_loss(output, data)
        errAE.backward()

        optimizerEnc.step() 
        optimizerDec.step()

        if i_batch % 50 == 0:
            print("Test Loss:")
            print("Epoch: ", epoch, " | i: ", i_batch)
            print("AutoEncoder Loss: ", errAE.item())
            
    ## Pass through networks and calculate losses
    errAE_Val = 0
    for i in val_ints:
        val_velocity = get_velocity_field(i)
        val_velocity = torch.from_numpy(val_velocity).unsqueeze(0).to(device=device, dtype=torch.float)

        # Pass through Encoder 
        val_output = netEnc(val_velocity).detach()
        val_output = netDec(val_output).detach()

        errAE_Val += mse_loss(val_output, val_velocity)

    if epoch % 200 == 0:
        print("Saving model at epoch ", epoch)
        torch.save({
                    'netEnc_state_dict': netEnc.state_dict(), 
                    'netDec_state_dict': netDec.state_dict(), 
                    'optimizerEnc_state_dict': optimizerEnc.state_dict(),
                    'optimizerDec_state_dict': optimizerDec.state_dict() },
                    "/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/vFinalAE" + str(epoch))
        print("Model has saved successfully!")

        # Save graph outputs
        newfilePath = '/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/GANgraphs/vAEFinal' + str(epoch) + '.csv'
        rows = zip(epoch_list, loss_list, val_loss_list)
        with open(newfilePath, "w") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
        
        
    errAE_Val /= len(val_ints)

    # Storing losses per epoch to plot
    epoch_list.append(epoch)
    loss_list.append(errAE.item())
    val_loss_list.append(errAE_Val.item())
    print("Validation Loss:", errAE_Val.item())

plt.plot(epoch_list, loss_list, label="Test Loss")
plt.plot(epoch_list, val_loss_list, label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Velocity AE Loss')
plt.legend()
plt.show()

# Save graph outputs
newfilePath = '/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/GANgraphs/vFinalAE1000.csv'
rows = zip(epoch_list, loss_list, val_loss_list)
with open(newfilePath, "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)


print("Training complete. Saving model...")
torch.save({
            'netEnc_state_dict': netEnc.state_dict(), 
            'netDec_state_dict': netDec.state_dict(), 
            'optimizerEnc_state_dict': optimizerEnc.state_dict(),
            'optimizerDec_state_dict': optimizerDec.state_dict() },
            "/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/vFinalAE1000")
print("Model has saved successfully!")
