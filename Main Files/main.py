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

#if __name__ == '__main__':

# Set random seed for reproducibility 
manualSeed = 9999
# manualSeed = random.ranint(1, 10000) # use if we want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)



########################
# Creating the dataset #
########################
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from getData import get_tracer

class TracerDataset(Dataset):
    """Tracer Dataset"""

    def __init__(self, file_number='', root_dir='', transform=None):
        """
        Initialise the Dataset
        :fileNumber: int or string
            Used to specify which file to open
        : rootDir: string
            Directory of all vtu files
        :transform: callable, optional
            Optional transform to be applied on a sample
        """
        #self.tracer_set = get_tracer(file_number)
        self.root_dir = root_dir
        self.transform = transform
        self.length = time_steps # number of timesteps available

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = get_tracer(idx)

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample):
        # swap axis because
        # numpy: H x W x C
        # torch: C x H x W
        
        #print("Size is ", sample.shape())
        #sample = sample.transpose((2, 0, 1)) # May need to flip this for Veloctiy
        sample = torch.from_numpy(sample)
        return sample

# tracer_dataset = TracerDataset()

# sample = tracer_dataset[988]
# # print(0, sample)
# # print("The size is ", sample.shape)
# to_tensor = ToTensor()
# sample = to_tensor(sample)
# # print(sample)
# # print(type(sample))
# # print(sample.shape)

tracer_dataset = TracerDataset(transform = ToTensor())
# for i in range(len(tracer_dataset)-1):
#     sample = tracer_dataset[i]

#     print(i, sample.size())

dataloader = DataLoader(tracer_dataset, batch_size=batch_size,
                            shuffle=True) # Should add workers


# Decide whihch device we want to run on
if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#print(device)
# Helper function to print out tracers
# def show_tracers_batch(sample_batched):
#     """Show tensors for tracers"""
#     batch_size = len(sample_batched)
#     im_size = tracer_input_size

#     for i in range(batch_size):
#         print(sample_batched[i])

# print(iter(dataloader))
# real_batch = next(iter(dataloader))

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched)

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
netG = Discriminator(ngpu).to(device)
print(netG)

# Handle multi-gpu if required
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights initialiser function to randomly initalise all weights
# to mean=0 and sd=0.2
netG.apply(weights_init)

for i_batch, sample_batched in enumerate(dataloader):
    data = sample_batched.to(device=device, dtype=torch.float)
    print(data.size())
    print("The size of the data before going in is ", data.size())
    # (Batch, No. Channels, Height, Width)
    # (16, 1, 10040, 1)
    # 
    # input: (N, Cin, L) and output: (N, Cout, Lout)
    # (Batch Size, No. Channels, Length)
    # (16, 1, 100040)
    netAE.zero_grad()
    outputAE = netAE(data)
    print("The size of the data after AutoEncoder is ", outputAE.size())
    print("")

    netG.zero_grad()
    outputG = netG(outputAE)
    print("The size of the data after Generator is ", outputG.size())
    print("")

    netD.zero_grad()
    outputD = netD(outputAE)
    print("The size of the data after Discriminator is ", outputD.size())