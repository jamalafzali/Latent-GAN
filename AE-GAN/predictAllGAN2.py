###############################################################
# Predict Tracers using GAN by continuously feeding itself in #
###############################################################

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from AutoEncoder import Encoder, Decoder
from Discriminator import Discriminator
from Generator import Generator
from Variables import *
from Dataset import *
from getData import *
from Norm import *
from nadam import Nadam
from createVTU import create_tracer_VTU

if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#################
# Instantiating #
#################
netEnc = Encoder(ngpu).to(device)
# netDec = Decoder(ngpu).to(device)
netD = Discriminator(ngpu).to(device)
netG = Generator(ngpu).to(device)

#################################
# Loss Functions and Optimisers #
#################################
# Setup Adam optimizers
optimizerEnc = Nadam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerDec = Nadam(netDec.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = Nadam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
#optimizerD = optim.SGD(netD.parameters(), lr=lr)

checkpoint = torch.load("E:/MSc Individual Project/Models/AutoEncoder64")
netEnc.load_state_dict(checkpoint['netEnc_state_dict'])
# netDec.load_state_dict(checkpoint['netDec_state_dict'])
optimizerEnc.load_state_dict(checkpoint['optimizerEnc_state_dict'])
# optimizerDec.load_state_dict(checkpoint['optimizerDec_state_dict'])

checkpoint = torch.load("E:/MSc Individual Project/Models/GAN128")
netG.load_state_dict(checkpoint['netG_state_dict'])
#netD.load_state_dict(checkpoint['netD_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
#optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

mse_loss = nn.MSELoss()

#tracer_dataset = TracerLatentDataset(transform = ToTensor())

batch_indicies = []
for i in range(988, 3729):
    batch_indicies.append([i])

#dataloader = DataLoader(tracer_dataset, batch_sampler=batch_indicies)

# for i_batch, sample_batched in enumerate(dataloader):
#     data = sample_batched.to(device=device, dtype=torch.float)
#     output = denormalise(netG(data), x_min, x_max)
#     output = np.array(output.squeeze().cpu().detach())

#     # Save output 
#     create_tracer_VTU(i_batch, output, "GAN")

# Read in 988
#batch_index = [988]
data = get_tracer(988)
data = torch.from_numpy(data).unsqueeze(1).to(device=device, dtype=torch.float)
output = netEnc(data)
output = denormalise(netG(output), x_min, x_max)
output = np.array(output.squeeze().cpu().detach())
# Save output
create_tracer_VTU(988, output, "GAN")
print("First is done!")

for i in range(989, 3729):
    #batch_index = [i-1]
    data = get_prediction_tracer(i-1)
    data = torch.from_numpy(data).unsqueeze(1).to(device=device, dtype=torch.float)
    output = netEnc(data)
    output = denormalise(netG(output), x_min, x_max)
    output = np.array(output.squeeze().cpu().detach())
    # Save output
    create_tracer_VTU(i, output, "GAN")
    print(i)
