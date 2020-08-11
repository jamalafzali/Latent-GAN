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
            nn.ConvTranspose1d(latent_size, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf*8),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True), # Try LeakyReLU, nn.LeakyReLU(0.2, inplace=True)

            nn.ConvTranspose1d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf*4),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),

            nn.ConvTranspose1d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf*2),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),

            nn.ConvTranspose1d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),

            nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        output = self.main(input)
        return output