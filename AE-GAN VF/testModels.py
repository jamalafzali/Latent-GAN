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

from AutoEncoder import Encoder, Decoder
from Discriminator import Discriminator
from Generator import Generator
from Variables import *
from Dataset import *
from getData import get_tracer
from Norm import *
from nadam import Nadam

if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#torch.set_default_dtype(torch.float64)

#################
# Instantiating #
#################
netEnc = Encoder(ngpu).to(device)
netDec = Decoder(ngpu).to(device)
#netD = Discriminator(ngpu).to(device)
#netG = Generator(ngpu).to(device)

#################################
# Loss Functions and Optimisers #
#################################
# Setup Adam optimizers
#optimizerEnc = optim.Nadam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999)) # Nadam optim?
#optimizerDec = optim.Adam(netDec.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerEnc = Nadam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDec = Nadam(netDec.parameters(), lr=lr, betas=(beta1, 0.999))

checkpoint = torch.load("E:\MSc Individual Project\Models\AutoEncoder64")

netEnc.load_state_dict(checkpoint['netEnc_state_dict'])
netDec.load_state_dict(checkpoint['netDec_state_dict'])

optimizerEnc.load_state_dict(checkpoint['optimizerEnc_state_dict'])
optimizerDec.load_state_dict(checkpoint['optimizerDec_state_dict'])


mse_loss = nn.MSELoss()

tracer_dataset = TracerDataset(transform = ToTensor())

batch_indicies = list(BatchSampler(RandomSampler(range(time_steps)), batch_size=batch_size, drop_last=True)) #Should include workers?


dataloader = DataLoader(tracer_dataset, batch_sampler=batch_indicies, num_workers=2) # Should add workers

print(batch_indicies)
for i_batch, (sample_batched, sample_batched_incr) in enumerate(zip(dataloader, dataloader_incr)):
    # Am I passing the same batches through each epoch?
    data = sample_batched.to(device=device, dtype=torch.float)
    data_incr = sample_batched.to(device=device, dtype=torch.float)
    print("i: ", i_batch)
    torch.set_printoptions(profile="full")
    output = denormalise(netDec(netEnc(data)), x_min, x_max)
    print(output)
    print("The loss of this batch is: ", mse_loss(output, denormalise(data_incr,x_min,x_max)).item())
