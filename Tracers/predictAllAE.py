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
from getData import get_tracer, get_prediction_tracer
from Norm import *
from nadam import Nadam
from createVTU import create_tracer_VTU_AE

filePathToModel = "E:/MSc Individual Project/Models/Experiments/FinalAE600"

if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#################
# Instantiating #
#################
netEnc = Encoder(ngpu).to(device)
netDec = Decoder(ngpu).to(device)

checkpoint = torch.load(filePathToModel)
netEnc.load_state_dict(checkpoint['netEnc_state_dict'])
netDec.load_state_dict(checkpoint['netDec_state_dict'])

mse_loss = nn.MSELoss()

batch_indicies = []
for i in range(3729):
    batch_indicies.append([i])

for i in batch_indicies:
    data = get_tracer(i)
    data = torch.from_numpy(data).unsqueeze(0).to(device=device, dtype=torch.float)
    output = denormalise(netDec(netEnc(data)), x_min, x_max)
    output = np.array(output.squeeze().cpu().detach())
    create_tracer_VTU_AE(i, output, "tAE")
    print(i)
