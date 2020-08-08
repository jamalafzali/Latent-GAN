#################################################
# Predict Tracers using GAN + Data Assimilation #
#################################################
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
netG = Generator(ngpu).to(device)

#####################
# Optimisers & Loss #
#####################
optimizerEnc = Nadam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = Nadam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
mse_loss = nn.MSELoss()

##################
# Loading Models #
##################
checkpoint = torch.load("E:/MSc Individual Project/Models/AutoEncoder64")
netEnc.load_state_dict(checkpoint['netEnc_state_dict'])
optimizerEnc.load_state_dict(checkpoint['optimizerEnc_state_dict'])

checkpoint = torch.load("E:/MSc Individual Project/Models/GAN128")
netG.load_state_dict(checkpoint['netG_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])

# Number of prediction tracers that will be used an inputs before using real data
numOfPreds = 1

i = 988
while i < 3729:
    # Take real input and output the prediction
    data = get_tracer(i)
    data = torch.from_numpy(data).unsqueeze(1).to(device=device, dtype=torch.float)
    output = netEnc(data)
    output = denormalise(netG(output), x_min, x_max)
    output = np.array(output.squeeze().cpu().detach())
    create_tracer_VTU(i, output, "GAN")
    i += 1

    # Take the prediction and output another prediction
    for _ in range(numOfPreds):
        data = get_prediction_tracer(i-1)
        data = torch.from_numpy(data).unsqueeze(1).to(device=device, dtype=torch.float)
        output = netEnc(data)
        output = denormalise(netG(output), x_min, x_max)
        output = np.array(output.squeeze().cpu().detach())
        create_tracer_VTU(i, output, "GAN")
        i += 1



