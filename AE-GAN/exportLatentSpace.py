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
from createVTU import create_tracer_VTU

if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#################
# Instantiating #
#################
netEnc = Encoder(ngpu).to(device)

#################################
# Loss Functions and Optimisers #
#################################
# Setup Adam optimizers
optimizerEnc = Nadam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999))

checkpoint = torch.load("E:/MSc Individual Project/Models/AutoEncoder64")

netEnc.load_state_dict(checkpoint['netEnc_state_dict'])
optimizerEnc.load_state_dict(checkpoint['optimizerEnc_state_dict'])


tracer_dataset = TracerDataset(transform = ToTensor())

batch_indicies = []
for i in range(3729):
    batch_indicies.append([i])
#batch_indicies_incr = [[i + 1 for i in item] for item in batch_indicies]


dataloader = DataLoader(tracer_dataset, batch_sampler=batch_indicies)#, num_workers=2) # Should add workers
#dataloader_incr = DataLoader(tracer_dataset, batch_sampler=batch_indicies_incr, num_workers=2)

# #print("netEnc datatype is: ", netEnc.dtype())
# weights = list(netEnc.parameters())[0].data.cpu().numpy()
# print(weights)
# print("The type of weights is: ", type(weights[0][0][0]))

#for i_batch, (sample_batched, sample_batched_incr) in enumerate(zip(dataloader, dataloader_incr)):
for i_batch, sample_batched in enumerate(dataloader):
    data = sample_batched.to(device=device, dtype=torch.float)
    #data_incr = sample_batched.to(device=device, dtype=torch.float)
    output = netEnc(data)
    output = np.array(output.squeeze().cpu().detach())

    #fileLocation = "/vol/bitbucket/ja819/Fluids Dataset/LatentSpace/LS_" + str(i_batch) +".csv"
    fileLocation = "E:/MSc Individual Project/Fluids Dataset/LatentSpace/LS_" + str(i_batch) +".csv"
    np.savetxt(fileLocation, output, delimiter=",")
    print(i_batch)


