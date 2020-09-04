import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from AutoEncoder import *
from Variables import *
from Dataset import *
from getData import get_tracer
from createVTU import create_tracer_VTU

if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#################
# Instantiating #
#################
netEnc = Encoder(ngpu).to(device)

# Location of the model to Load
checkpoint = torch.load("E:/MSc Individual Project/Models/AutoEncoder64")

netEnc.load_state_dict(checkpoint['netEnc_state_dict'])

tracer_dataset = TracerDataset(transform = ToTensor())

batch_indicies = []
for i in range(3729):
    batch_indicies.append([i])

dataloader = DataLoader(tracer_dataset, batch_sampler=batch_indicies)

for i_batch, sample_batched in enumerate(dataloader):
    data = sample_batched.to(device=device, dtype=torch.float)
    output = netEnc(data)
    output = np.array(output.squeeze().cpu().detach())

    fileLocation = defaultFilePath + "/LatentSpace/LS_" + str(i_batch) +".csv"
    np.savetxt(fileLocation, output, delimiter=",")
    print(i_batch)