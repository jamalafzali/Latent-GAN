import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from Variables import *

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose1d(latent_size, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True), # Try LeakyReLU, nn.LeakyReLU(0.2, inplace=True)

            nn.ConvTranspose1d(ndf*8, ndf*4, 4, 2, 1, 1, bias=False),
            nn.BatchNorm1d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),

            nn.ConvTranspose1d(ndf*4, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),

            nn.ConvTranspose1d(ndf*2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),

            nn.ConvTranspose1d(ndf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        output = self.main(input)
        return output