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
from getData import get_tracer
from Norm import *
from nadam import Nadam
from createVTU import create_tracer_VTU_GAN

filePathToModelAE = "E:/MSc Individual Project/Models/Experiments/FinalAE600"
filePathToModelGAN = "E:/MSc Individual Project/Models/Experiments/FinalGAN400"

if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#################
# Instantiating #
#################
netEnc = Encoder(ngpu).to(device)
netG = Generator(ngpu).to(device)

checkpoint = torch.load(filePathToModelAE)
netEnc.load_state_dict(checkpoint['netEnc_state_dict'])

checkpoint = torch.load(filePathToModelGAN)
netG.load_state_dict(checkpoint['netG_state_dict'])

mse_loss = nn.MSELoss()

tracer_dataset = TracerDataset(transform = ToTensor())

batch_indicies = []
for i in range(3729):
    batch_indicies.append(i)

for i in batch_indicies:
    data = get_tracer(i)
    data = torch.from_numpy(data).unsqueeze(0).to(device=device, dtype=torch.float)
    output = denormalise(netG(netEnc(data)), x_min, x_max)
    output = np.array(output.squeeze().cpu().detach())
    create_tracer_VTU_GAN(i, output, "tGAN")
    print(i)
