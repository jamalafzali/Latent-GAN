import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from Variables import *

# Tracer input will always be (1 x 100040)
class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv1d(nc, naef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(naef),
            
            nn.Conv1d(naef, naef*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(naef*2),

            nn.Conv1d(naef*2, naef*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(naef*4),

            nn.Conv1d(naef*4, naef*8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(naef*8),

            nn.Conv1d(naef*8, latent_size, 4, 2, 1, bias=False),
        )
    
    def forward(self, input):
        output = self.main(input)
        return output


class Decoder(nn.Module):
    def __init__(self, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose1d(latent_size, naef*8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(naef*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(naef*8, naef*4, 4, 2, 1, 1, bias=False),
            nn.BatchNorm1d(naef*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(naef*4, naef*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(naef*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(naef*2, naef, 4, 2, 1, bias=False),
            nn.BatchNorm1d(naef),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose1d(naef, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        output = self.main(input)
        return output