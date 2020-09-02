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
from createVTU import create_tracer_VTU_GAN

if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#################
# Instantiating #
#################
netEnc = Encoder(ngpu).to(device)
netG = Generator(ngpu).to(device)

mse_loss = nn.MSELoss()

##################
# Loading Models #
##################
checkpoint = torch.load("E:/MSc Individual Project/Models/Experiments/FinalAE600")
netEnc.load_state_dict(checkpoint['netEnc_state_dict'])

checkpoint = torch.load("E:/MSc Individual Project/Models/Experiments/GANFinal400")
netG.load_state_dict(checkpoint['netG_state_dict'])


# Number of prediction tracers that will be used an inputs before using real data
numOfPreds = 1

i = 988
# Take real 988 and output the prediction to 989
data = get_tracer(i)
data = torch.from_numpy(data).unsqueeze(1).to(device=device, dtype=torch.float)
output = netEnc(data)
output = denormalise(netG(output), x_min, x_max)
output = np.array(output.squeeze().cpu().detach())
create_tracer_VTU_GAN(i, output, "tDA")
i += 1

# Values for weighted average
a = 0.1 # Weight for prediction 0.1
b = 0.9 # Weight for real 0.9

while i < 3729:
    # Take the real input and the prediction input,
    # and take an average of both, 
    # then output the prediction
    data = get_prediction_tracer(i-1)
    data = torch.from_numpy(data).unsqueeze(1).to(device=device, dtype=torch.float)
    output = netEnc(data)
    output = denormalise(netG(output), x_min, x_max)
    output = np.array(output.squeeze().cpu().detach())

    real = get_tracer(i+1)
    real = real[0,:]
    
    # Perform weighted average
    assim = (a*output + b*real) / (a+b) # Probably better ways to do this

    create_tracer_VTU_GAN(i, assim, "tDA")
    i += 1

    # Take the prediction and output another prediction
    for _ in range(numOfPreds):
        pred = get_prediction_tracer(i-1)
        pred = torch.from_numpy(pred).unsqueeze(1).to(device=device, dtype=torch.float)
        output = netEnc(pred)
        output = denormalise(netG(output), x_min, x_max)
        output = np.array(output.squeeze().cpu().detach())
        create_tracer_VTU_GAN(i, output, "tDA")
        i += 1

