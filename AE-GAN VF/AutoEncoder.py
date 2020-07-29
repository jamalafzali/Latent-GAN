import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from Variables import *

# Input will always be 1 X 100040
class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv1d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.5),
            #nn.ReLU(True), # Try LeakyReLU nn.LeakyReLU(0.2, inplace=True)
            nn.BatchNorm1d(ndf),
            # 
            nn.Conv1d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.5),
            #nn.ReLU(True),
            nn.BatchNorm1d(ndf*2),

            nn.Conv1d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.5),
            #nn.ReLU(True),
            nn.BatchNorm1d(ndf*4),

            nn.Conv1d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.5),
            #nn.ReLU(True),
            nn.BatchNorm1d(ndf*8),

            nn.Conv1d(ndf*8, latent_size, 4, 2, 1, bias=False),
            #nn.MaxPool1d(4,2,1)
        )
    
    def forward(self, input):
        output = self.main(input)
        return output


class Decoder(nn.Module):
    def __init__(self, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose1d(latent_size, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf*8),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True), # Try LeakyReLU, nn.LeakyReLU(0.2, inplace=True)

            nn.ConvTranspose1d(ndf*8, ndf*4, 4, 2, 1, 1, bias=False),
            nn.BatchNorm1d(ndf*4),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),

            nn.ConvTranspose1d(ndf*4, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf*2),
           #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(True),

            nn.ConvTranspose1d(ndf*2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True), 
            #nn.ReLU(True), # Try with this again

            nn.ConvTranspose1d(ndf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        output = self.main(input)
        return output