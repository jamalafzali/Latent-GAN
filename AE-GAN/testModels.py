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
netD = Discriminator(ngpu).to(device)
netG = Generator(ngpu).to(device)

#################################
# Loss Functions and Optimisers #
#################################
# Setup Adam optimizers
#optimizerEnc = optim.Nadam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999)) # Nadam optim?
#optimizerDec = optim.Adam(netDec.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerEnc = Nadam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDec = Nadam(netDec.parameters(), lr=lr, betas=(beta1, 0.999))

checkpoint = torch.load("/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/AutoEncoder64")
#checkpoint = torch.load("/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/GAN")

netEnc.load_state_dict(checkpoint['netEnc_state_dict'])
netDec.load_state_dict(checkpoint['netDec_state_dict'])
# netG.load_state_dict(checkpoint['netG_state_dict'])
# netD.load_state_dict(checkpoint['netD_state_dict'])
optimizerEnc.load_state_dict(checkpoint['optimizerEnc_state_dict'])
optimizerDec.load_state_dict(checkpoint['optimizerDec_state_dict'])
# optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
# optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

# torch.save({
#             'netEnc_state_dict': netEnc.state_dict(), 
#             'netG_state_dict': netG.state_dict(), 
#             'netD_state_dict': netD.state_dict(),
#             'optimizerEnc_state_dict': optimizerEnc.state_dict(),
#             'optimizerG_state_dict': optimizerG.state_dict(),
#             'optimizerD_state_dict': optimizerD.state_dict() },
#             "/vol/bitbucket/ja819/Python Files/Latent-GAN/Main Files/Saved models/GAN")

mse_loss = nn.MSELoss()

tracer_dataset = TracerDataset(transform = ToTensor())

batch_indicies = list(BatchSampler(RandomSampler(range(time_steps)), batch_size=batch_size, drop_last=True)) #Should include workers?
#batch_indicies = list(BatchSampler())
#batch_indicies_incr = [[i + 1 for i in item] for item in batch_indicies]


dataloader = DataLoader(tracer_dataset, batch_sampler=batch_indicies, num_workers=2) # Should add workers
#dataloader_incr = DataLoader(tracer_dataset, batch_sampler=batch_indicies_incr, num_workers=2)

print(repr(netEnc))
print(netEnc)

# #print("netEnc datatype is: ", netEnc.dtype())
# weights = list(netEnc.parameters())[0].data.cpu().numpy()
# print(weights)
# print("The type of weights is: ", type(weights[0][0][0]))

# print(batch_indicies)
# for i_batch, (sample_batched, sample_batched_incr) in enumerate(zip(dataloader, dataloader_incr)):
#     # Am I passing the same batches through each epoch?
#     data = sample_batched.to(device=device, dtype=torch.float)
#     data_incr = sample_batched.to(device=device, dtype=torch.float)
#     print("i: ", i_batch)
#     torch.set_printoptions(profile="full")
#     #print(denormalise(data, x_min, x_max))
#     # print("Data:")
#     # print(data[0][0][34232].item())
#     # print(data[0][0][34232].type())
#     #print("Normalised output:")
#     #print(netDec(netEnc(data)), x_min, x_max)
#     #print("Denormalised output:")
#     output = denormalise(netDec(netEnc(data)), x_min, x_max)
#     print(output)
#     # print(denormalise(netG(netEnc(data)), x_min, x_max))
#     # torch.set_printoptions(profile="default")
#     print("The loss of this batch is: ", mse_loss(output, denormalise(data_incr,x_min,x_max)).item())
#     break


# # p = get_tracer(20)

# # print(netG(netAE(p)))

# # netAE.eval()
# # netG.eval()
# # netD.eval()
# # # - or -
# # netAE.train()
# # netG.train()
# # netD.train()