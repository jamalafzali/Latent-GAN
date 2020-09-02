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
from getData import get_velocity_field
from Norm import *
from nadam import Nadam
from createVTU import create_velocity_field_VTU_GAN

if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#################
# Instantiating #
#################
netEnc = Encoder(ngpu).to(device)
netG = Generator(ngpu).to(device)

#################################
# Loss Functions and Optimisers #
#################################
# Setup Adam optimizers
optimizerEnc = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
#optimizerDec = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
#optimizerD = optim.SGD(netD.parameters(), lr=lr)

#checkpoint = torch.load("E:/MSc Individual Project/Models/Experiments/AE256")
checkpoint = torch.load("E:/MSc Individual Project/Models/Experiments/vFinalAEFinal")
netEnc.load_state_dict(checkpoint['netEnc_state_dict'])
#netDec.load_state_dict(checkpoint['netDec_state_dict'])
optimizerEnc.load_state_dict(checkpoint['optimizerEnc_state_dict'])
#optimizerDec.load_state_dict(checkpoint['optimizerDec_state_dict'])

checkpoint = torch.load("E:/MSc Individual Project/Models/Experiments/vGANFinal_200")
netG.load_state_dict(checkpoint['netG_state_dict'])
#netD.load_state_dict(checkpoint['netD_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
#optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

# print(netG)
# print(optimizerG)
# print(netD)
# print(optimizerD)

mse_loss = nn.MSELoss()

tracer_dataset = VelocityFieldDataset(transform = ToTensor())


batch_indicies = []
for i in range(900, 989):
    batch_indicies.append(i)

for i in batch_indicies:
    data = get_velocity_field(i)
    data = torch.from_numpy(data).unsqueeze(0).to(device=device, dtype=torch.float)
    output = denormalise(netG(netEnc(data)), x_min, x_max)
    output = np.array(output.squeeze().cpu().detach()).transpose()
    print(output.shape)
    create_velocity_field_VTU_GAN(i, output, "vGANFinal_200")
    print(i)