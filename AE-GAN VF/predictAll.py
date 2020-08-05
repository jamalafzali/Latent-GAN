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
from createVTU import create_velocity_field_VTU

if torch.cuda.is_available():
    print("CUDA is available!")
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#################
# Instantiating #
#################
netEnc = Encoder(ngpu).to(device)
netDec = Decoder(ngpu).to(device)
# netD = Discriminator(ngpu).to(device)
# netG = Generator(ngpu).to(device)

#################################
# Loss Functions and Optimisers #
#################################
# Setup Adam optimizers
optimizerEnc = Nadam(netEnc.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDec = Nadam(netDec.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

checkpoint = torch.load("E:/MSc Individual Project/Models/AutoEncoderVF200")
netEnc.load_state_dict(checkpoint['netEnc_state_dict'])
netDec.load_state_dict(checkpoint['netDec_state_dict'])
optimizerEnc.load_state_dict(checkpoint['optimizerEnc_state_dict'])
optimizerDec.load_state_dict(checkpoint['optimizerDec_state_dict'])

# checkpoint = torch.load("E:/MSc Individual Project/Models/GAN64")
# netG.load_state_dict(checkpoint['netG_state_dict'])
# netD.load_state_dict(checkpoint['netD_state_dict'])
# optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
# optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])


mse_loss = nn.MSELoss()

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

#print(batch_indicies)
#for i_batch, (sample_batched, sample_batched_incr) in enumerate(zip(dataloader, dataloader_incr)):
for i_batch, sample_batched in enumerate(dataloader):
    data = sample_batched.to(device=device, dtype=torch.float)
    #data_incr = sample_batched.to(device=device, dtype=torch.float)
    output = denormalise(netDec(netEnc(data)), x_min, x_max)
    #output = denormalise(netDec(netEnc(data)), x_min, x_max)
    output = np.array(output.squeeze().cpu().detach()).transpose()
    #print("The shape is : ",output.shape)
    create_velocity_field_VTU(i_batch, output, "AEVF")
    print(i_batch)
    #print(np.array(output.squeeze().cpu().detach()))
    # print(denormalise(netG(netEnc(data)), x_min, x_max))
    # torch.set_printoptions(profile="default")
    #print("The loss of this batch is: ", mse_loss(output, denormalise(data_incr,x_min,x_max)).item())
    #break


