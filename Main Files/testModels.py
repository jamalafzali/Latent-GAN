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

#import pandas as pd
#from skimage import io, transform

from torchvision import transforms, utils

from AutoEncoder import AutoEncoder
from Discriminator import Discriminator
from Generator import Generator
from Variables import *
from Dataset import *
from getData import get_tracer

if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


#################
# Instantiating #
#################
netAE = AutoEncoder(ngpu).to(device)
netD = Discriminator(ngpu).to(device)
netG = Generator(ngpu).to(device)

#################################
# Loss Functions and Optimisers #
#################################
# Setup Adam optimizers
optimizerAE = optim.Adam(netAE.parameters(), lr=lr, betas=(beta1, 0.999)) # Nadam optim?
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))


checkpoint = torch.load("/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/test")
netAE.load_state_dict(checkpoint['netAE_state_dict'])
netG.load_state_dict(checkpoint['netG_state_dict'])
netD.load_state_dict(checkpoint['netD_state_dict'])
optimizerAE.load_state_dict(checkpoint['optimizerAE_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

tracer_dataset = TracerDataset(transform = ToTensor())

batch_indicies = list(BatchSampler(RandomSampler(range(time_steps)), batch_size=batch_size, drop_last=True)) #Should include workers?
batch_indicies_incr = [[i + 1 for i in item] for item in batch_indicies]

dataloader = DataLoader(tracer_dataset, batch_sampler=batch_indicies, num_workers=2) # Should add workers
dataloader_incr = DataLoader(tracer_dataset, batch_sampler=batch_indicies_incr, num_workers=2)

print(batch_indicies)
for i_batch, (sample_batched, sample_batched_incr) in enumerate(zip(dataloader, dataloader_incr)):
    # Am I passing the same batches through each epoch?
    data = sample_batched.to(device=device, dtype=torch.float)
    data_incr = sample_batched.to(device=device, dtype=torch.float)
    print("i: ", i_batch)
    torch.set_printoptions(profile="full")
    print(netG(netAE(data)))
    torch.set_printoptions(profile="default")

# p = get_tracer(20)

# print(netG(netAE(p)))

netAE.eval()
netG.eval()
netD.eval()
# - or -
netAE.train()
netG.train()
netD.train()