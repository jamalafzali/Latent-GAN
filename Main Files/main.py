from __future__ import print_function, division
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import pandas as pd
from skimage import io, transform

from torchvision import transforms, utils

from AutoEncoder import AutoEncoder
from Discriminator import Discriminator
from Generator import Generator
from Variables import *
from Dataset import *
from getData import get_tracer

#if __name__ == '__main__':

# Set random seed for reproducibility 
manualSeed = 9999
# manualSeed = random.ranint(1, 10000) # use if we want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#########################
# Instantiating Dataset #
#########################
tracer_dataset = TracerDataset(transform = ToTensor())

# Creating list of batches alongside an increment of this (for piece-wise error calculations)
batch_indicies = list(BatchSampler(RandomSampler(range(time_steps)), batch_size=batch_size, drop_last=True)) #Should include workers?
batch_indicies_incr = [[i + 1 for i in item] for item in batch_indicies]

dataloader = DataLoader(tracer_dataset, batch_sampler=batch_indicies) # Should add workers
dataloader_incr = DataLoader(tracer_dataset, batch_sampler=batch_indicies_incr)

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


#############################
# Instantiating AutoEncoder #
#############################
netAE = AutoEncoder(ngpu).to(device)
print(netAE)

# Handle multi-gpu if required
if (device.type == 'cuda') and (ngpu > 1):
    netAE = nn.DataParallel(netAE, list(range(ngpu)))

# Apply the weights initialiser function to randomly initalise all weights
# to mean=0 and sd=0.2
netAE.apply(weights_init)


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
bce_loss = nn.BCELoss()
# *** Physics based Loss *** #

# Labels for training
real_label = 1
fake_label = 0

# Setup Adam optimizers
optimizerAE = optim.Adam(netAE.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

############
# Training #
############

for i_batch, (sample_batched, sample_batched_incr) in enumerate(zip(dataloader, dataloader_incr)):
    data = sample_batched.to(device=device, dtype=torch.float)
    data_incr = sample_batched.to(device=device, dtype=torch.float)

    #################################################################
    # (1) Update Discriminator: maximise log(D(x)) + log(1-D(G(z))) #
    #################################################################

    ## Training with the all-real batch
    netD.zero_grad()
    #real_data = data[0].to(device)
    label = torch.full((batch_size,), real_label, device=device, dtype = torch.float)
    print("Label is: ", label)
    print("Label size is : ", label.size())
    print("Data size is: ", data.size())
    # Forward pass through Discriminator
    outputD = netD(data)
    print("The size of the data after Discriminator is ", outputD.size())    
    # Calculate loss
    errD_real = bce_loss(outputD, label)
    print("errD_real is: ", errD_real)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    

    # #print(data.size())
    # print("The size of the data before going in is ", data.size())
    # # (Batch, No. Channels, Height, Width)
    # # (16, 1, 10040, 1)
    # # 
    # # input: (N, Cin, L) and output: (N, Cout, Lout)
    # # (Batch Size, No. Channels, Length)
    # # (16, 1, 100040)
    # netAE.zero_grad()
    # outputAE = netAE(data)
    # print("The size of the data after AutoEncoder is ", outputAE.size())
    # print("")

    # netG.zero_grad()
    # outputG = netG(outputAE)
    # print("The size of the data after Generator is ", outputG.size())
    # print("")

    # netD.zero_grad()
    # outputD = netD(outputAE)
